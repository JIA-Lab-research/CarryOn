"""Route model requests from an isolated recovery worker without retries.

The run controller shares one budget directory across all episodes."""
from __future__ import annotations
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import functools
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

SPECS={
    "planner":("qwen/qwen3-vl-32b-instruct",0.2,4096,None),
    "judge":("qwen/qwen3-vl-32b-instruct",0.,512,None),
    "profile_summary":("qwen/qwen3-vl-32b-instruct",0.,512,None),
    "reground_selector":("qwen/qwen3-vl-235b-a22b-instruct",0.2,4096,None),
    "pose_classifier":("qwen/qwen3-vl-235b-a22b-instruct",0.,700,0.8),
}

class ResourceLimit(RuntimeError):
    pass


def write(path,value):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    temp=path.with_name(path.name+".tmp."+str(os.getpid())+"."+str(threading.get_ident()))
    temp.write_bytes(data);os.replace(temp,path)


def valid_route(route):
    parsed=urlsplit(route["endpoint"])
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Model endpoint must not embed credentials or URL parameters")
    local=route.get("transport")=="local"
    if local:
        if parsed.scheme!="http" or parsed.hostname not in ("127.0.0.1","localhost","::1"):
            raise ValueError("Local model endpoint must use loopback HTTP")
    else:
        if parsed.scheme!="https" or not route.get("api_key_env"):
            raise ValueError("Remote model route requires HTTPS and an explicit credential environment name")
        provider=route.get("provider",{})
        if (provider.get("allow_fallbacks") is not False or provider.get("data_collection")!="deny"
                or not isinstance(provider.get("only"),list) or len(provider["only"])!=1
                or not route.get("response_provider_aliases")):
            raise ValueError("Remote model route must name one provider and forbid fallback/data collection")
    ceiling=Decimal(str(route.get("per_request_ceiling_usd",0)))
    if not ceiling.is_finite() or ceiling<0 or (not local and ceiling<=0):
        raise ValueError("Invalid per-request cost ceiling")
    return local,ceiling


class Sender:
    def __init__(self,config,archive,row_id,*,post=None):
        self.config=config;self.archive=Path(archive).resolve();self.row_id=row_id
        self.budget=Path(config["budget_directory"]).resolve()
        self.maximum=config["max_requests"];self.maximum_paid=config.get("max_paid_requests",self.maximum);self.cap=Decimal(str(config["budget_usd"]))
        if (type(self.maximum)is not int or self.maximum<0 or type(self.maximum_paid)is not int
                or self.maximum_paid<0 or self.maximum_paid>self.maximum or not self.cap.is_finite() or self.cap<0):
            raise ValueError("Invalid shared model budget")
        if not config.get("routes") or set(config["routes"])-set(SPECS):raise ValueError("Unknown model role")
        self.routes=config["routes"]
        for route in self.routes.values():valid_route(route)
        self.counts={};self.lock=threading.Lock();self.post=post
        self.archive.mkdir(parents=True,exist_ok=False)
        self.budget.mkdir(parents=True,exist_ok=True)
        # Binding prevents a later episode from resetting or raising the same ledger's limits.
        signature=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        with self.locked():
            p=self.budget/"STATE.json"
            if p.exists():
                state=json.loads(p.read_text())
                if state["config_sha256"]!=signature:raise ValueError("Model budget configuration changed")
            else:write(p,{"config_sha256":signature,"attempts":0,"paid_attempts":0,"cost_upper_bound_usd":"0","halted":False,"calls":{}})

    @contextmanager
    def locked(self):
        if os.name!="posix":raise RuntimeError("Recovery shared-budget execution requires Linux")
        import fcntl
        with (self.budget/"LOCK").open("a") as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)

    def send(self,role,body):
        try:return self._send(role,body)
        except BaseException as exc:
            with self.lock:
                with (self.archive/"REJECTIONS.jsonl").open("a") as stream:
                    stream.write(json.dumps({"role":role,"error":type(exc).__name__,
                        "resource_limit":isinstance(exc,ResourceLimit)})+"\n")
            raise

    def _send(self,role,body):
        if role not in self.routes or role not in SPECS:raise ValueError("Unconfigured model role: "+role)
        model,temp,tokens,top_p=SPECS[role]
        if body.get("model")!=model or body.get("temperature")!=temp or body.get("max_tokens")!=tokens:
            raise ValueError("Frozen model or sampling envelope changed: "+role)
        if (top_p is None and "top_p" in body) or (top_p is not None and body.get("top_p")!=top_p):
            raise ValueError("Frozen top_p envelope changed: "+role)
        route=self.routes[role];local,ceiling=valid_route(route)
        headers={"Content-Type":"application/json"}
        if route.get("api_key_env"):
            key=os.environ.get(route["api_key_env"])
            if not key:raise ValueError("Missing credential environment: "+route["api_key_env"])
            headers["Authorization"]="Bearer "+key
        payload=dict(body)
        if "provider" in route:payload["provider"]=route["provider"]
        elif local:payload.pop("provider",None)
        raw=json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
        with self.lock:
            ordinal=self.counts.get(role,0)+1
            limit=self.config["per_episode_limits"][role]
            if limit is not None and ordinal>limit:raise ResourceLimit("Episode model role cap: "+role)
            self.counts[role]=ordinal
            with self.locked():
                state=json.loads((self.budget/"STATE.json").read_text());reserved=Decimal(0) if local else ceiling
                if (state["halted"] or state["attempts"]>=self.maximum or
                        (not local and state["paid_attempts"]>=self.maximum_paid) or
                        Decimal(state["cost_upper_bound_usd"])+reserved>self.cap):
                    raise ResourceLimit("Shared request or cost cap reached before send")
                call=state["attempts"]+1;state["attempts"]=call
                state["paid_attempts"]+=int(not local)
                state["cost_upper_bound_usd"]=str(Decimal(state["cost_upper_bound_usd"])+reserved)
                state["calls"][str(call)]={"row_id":self.row_id,"role":role,"ordinal":ordinal,"reserved_usd":str(reserved),"status":"reserved"}
                write(self.budget/"STATE.json",state)
        folder=self.archive/("call_%05d"%call);folder.mkdir()
        (folder/"request.body.bin").write_bytes(raw)
        record={"role":role,"ordinal":ordinal,"global_call":call,"transport":"local" if local else "remote","request_sha256":hashlib.sha256(raw).hexdigest(),
                "endpoint_host":urlsplit(route["endpoint"]).hostname,"reserved_usd":str(reserved),"automatic_retries":0,
                "started":time.time(),"status":"reserved","fixture":self.post is not None}
        write(folder/"ATTEMPT.json",record)
        actual=None;halt=False
        try:
            if self.post is None:
                import httpx
                with httpx.Client(timeout=route.get("timeout_s",180),trust_env=False,follow_redirects=False,
                                  transport=httpx.HTTPTransport(retries=0)) as client:
                    response=client.post(route["endpoint"],content=raw,headers=headers)
                    status,data=response.status_code,response.content
            else:status,data=self.post(route["endpoint"],raw,headers)
            (folder/"response.body.bin").write_bytes(data)
            record.update(http_status=status,response_sha256=hashlib.sha256(data).hexdigest())
            if status!=200:raise RuntimeError("Model service HTTP "+str(status))
            value=json.loads(data)
            if value.get("model") not in {model,*route.get("response_model_aliases",[])}:
                raise ValueError("Returned model identity differs")
            if not local and value.get("provider") not in route["response_provider_aliases"]:
                raise ValueError("Returned provider identity differs")
            content=value["choices"][0]["message"]["content"]
            if not isinstance(content,str) or not content.strip():raise ValueError("Empty model response")
            record["usage"]=value.get("usage")
            cost=(value.get("usage") or {}).get("cost")
            if cost is not None and not local:
                actual=Decimal(str(cost))
                if not actual.is_finite() or actual<0:actual=None;raise ValueError("Invalid service cost")
                if actual>ceiling:halt=True;raise ResourceLimit("Service exceeded the reserved ceiling; further calls stopped")
            record["status"]="completed"
            return status,data
        except BaseException as exc:
            record.update(status="resource_truncated" if isinstance(exc,ResourceLimit) else "service_error",error=type(exc).__name__)
            raise
        finally:
            with self.locked():
                state=json.loads((self.budget/"STATE.json").read_text())
                if actual is not None:state["cost_upper_bound_usd"]=str(Decimal(state["cost_upper_bound_usd"])+actual-reserved)
                state["halted"]=state["halted"] or halt
                state["calls"][str(call)].update(status=record["status"],response_archive=str(folder),actual_cost_usd=str(actual) if actual is not None else None)
                write(self.budget/"STATE.json",state)
            write(folder/"RESULT.json",record)

    def pose(self,raw):
        return self.send("pose_classifier",json.loads(raw))

    def install(self):
        from carryon.providers import openrouter as module
        from carryon.memory.user_summarizer import UserSummarizer
        sender=self
        # The profile summarizer shares the judge sampling settings.
        # Bind its call context explicitly; ContextVar isolates concurrent calls.
        context_role=ContextVar("carryon_recovery_model_role",default=None)
        original_summary=UserSummarizer.maybe_summarize
        @functools.wraps(original_summary)
        async def summarize(instance,*args,**kwargs):
            token=context_role.set("profile_summary")
            try:return await original_summary(instance,*args,**kwargs)
            finally:context_role.reset(token)
        UserSummarizer.maybe_summarize=summarize
        async def chat(client,*,model,messages,temperature=.2,max_tokens=4096,response_format=None):
            explicit=context_role.get()
            roles=([explicit] if explicit is not None else
                   [r for r,s in SPECS.items() if r!="profile_summary" and s[:3]==(model,temperature,max_tokens)])
            if len(roles)!=1 or roles[0]=="pose_classifier":raise ValueError("Unbound chat role")
            body=dict(model=model,messages=messages,temperature=temperature,max_tokens=max_tokens)
            if response_format is not None:body["response_format"]=response_format
            _,data=await asyncio.to_thread(sender.send,roles[0],body)
            value=json.loads(data);return module.ChatResult(text=value["choices"][0]["message"]["content"],raw=value)
        module.OpenRouterChatClient.chat=chat
