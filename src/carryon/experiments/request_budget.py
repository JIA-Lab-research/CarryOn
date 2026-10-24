"""Count and cap native requests.post calls inside one isolated experiment worker."""
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import threading
from .files import read_json,write_json


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


class BudgetedPost:
    def __init__(self,config,archive,post):
        if os.name!="posix":raise ValueError("Shared request budgets require Linux")
        self.config=config;self.archive=Path(archive);self.post=post
        self.root=Path(config["budget_directory"]);self.root.mkdir(parents=True,exist_ok=True)
        self.archive.mkdir(parents=True,exist_ok=False)
        self.cap=Decimal(str(config["budget_usd"]));self.ceiling=Decimal(str(config["per_request_ceiling_usd"]))
        if (type(config["max_attempts"]) is not int or config["max_attempts"]<0 or
                not self.cap.is_finite() or not self.ceiling.is_finite() or self.ceiling<=0 or self.cap<0):
            raise ValueError("Invalid request budget")
        self.signature=digest(config)
        with self.lock():
            path=self.root/"STATE.json"
            if path.exists():
                if read_json(path)["config_sha256"]!=self.signature:raise ValueError("Budget configuration changed")
            else:self.save({"config_sha256":self.signature,"attempts":0,"cost_upper_bound_usd":"0","halted":False})

    @contextmanager
    def lock(self):
        import fcntl
        with (self.root/"LOCK").open("a") as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)

    def save(self,state):
        path=self.root/("STATE.tmp."+str(os.getpid())+"."+str(threading.get_ident()))
        write_json(path,state);os.replace(path,self.root/"STATE.json")

    def __call__(self,url,**kwargs):
        payload=kwargs.get("json");cfg=self.config
        if url!=cfg["endpoint"] or not isinstance(payload,dict):raise ValueError("Unconfigured model request")
        fixed={k:payload.get(k) for k in ("model","provider","temperature","max_tokens")}
        if fixed!=cfg["envelope"]:raise ValueError("Model sampling/routing differs from the admitted envelope")
        messages=payload.get("messages")
        if (set(payload)!={"model","provider","temperature","max_tokens","messages"} or
                not isinstance(messages,list) or len(messages)!=2 or messages[0].get("role")!="system" or messages[1].get("role")!="user" or
                not isinstance(messages[0].get("content"),str) or not isinstance(messages[1].get("content"),str) or
                digest(messages[0]["content"])!=cfg["system_prompt_sha256"] or
                digest(messages[1]["content"]) not in cfg["allowed_user_message_sha256"]):
            raise ValueError("Request content is outside the admitted instruction-only scope")
        with self.lock():
            state=read_json(self.root/"STATE.json")
            if state["halted"] or state["attempts"]>=cfg["max_attempts"] or Decimal(state["cost_upper_bound_usd"])+self.ceiling>self.cap:
                raise RuntimeError("Shared model request/cost cap reached before sending")
            state["attempts"]+=1;ordinal=state["attempts"]
            state["cost_upper_bound_usd"]=str(Decimal(state["cost_upper_bound_usd"])+self.ceiling);self.save(state)
        folder=self.archive/("call_%05d"%ordinal);folder.mkdir()
        write_json(folder/"REQUEST.json",payload)
        record={"global_attempt":ordinal,"request_sha256":digest(payload),"reserved_usd":str(self.ceiling),"status":"reserved","automatic_retries":0}
        write_json(folder/"ATTEMPT.json",record)
        actual=None;halt=False
        try:
            response=self.post(url,**dict(kwargs,allow_redirects=False))
            (folder/"response.body.bin").write_bytes(response.content)
            record.update(status="completed" if response.status_code==200 else "http_error",http_status=response.status_code)
            if response.status_code==200:
                value=response.json()
                if value.get("model") not in {cfg["envelope"]["model"],*cfg.get("response_model_aliases",[])} or value.get("provider") not in cfg["response_provider_aliases"]:
                    raise ValueError("Returned model/provider identity differs")
                usage=value.get("usage") or {};record["usage"]=usage
                if usage.get("cost") is not None:
                    actual=Decimal(str(usage["cost"]))
                    if not actual.is_finite() or actual<0:actual=None;raise ValueError("Invalid service cost")
                    if actual>self.ceiling:halt=True;raise RuntimeError("Response exceeded reserved cost; further calls stopped")
            # Preserve the native caller's HTTP status handling and retry policy.
            return response
        except BaseException as exc:
            record.update(status="service_error",error=type(exc).__name__);raise
        finally:
            with self.lock():
                state=read_json(self.root/"STATE.json")
                if actual is not None:state["cost_upper_bound_usd"]=str(Decimal(state["cost_upper_bound_usd"])+actual-self.ceiling)
                state["halted"]|=halt;self.save(state)
            write_json(folder/"RESULT.json",record)
