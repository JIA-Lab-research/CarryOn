"""Execute three rounds of recovery with reflection and failed-case retries."""
from pathlib import Path
from decimal import Decimal
from ..files import read_json, write_json
from ..reflection import memory, collect, prepare, transport


def reflect(round_output, parent, output, *, suite, generation, model_config):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    collect.collect(round_output,parent,output/"evidence",suite=suite,generation=generation)
    requests=prepare.prepare(output/"evidence",parent,model_config,output/"requests")
    # send() returns an empty response list before constructing any client when
    # no evidence is eligible; no credentials are needed for inheritance.
    responses=transport.send(output/"requests",model_config,output/"responses")
    state=memory.publish(parent,requests,responses,output/"memory")
    state["reflection_cost"]=read_json(output/"responses/COST.json")
    return state


def execute(cases, output, *, suite, run_round,
            model_config=None, reflect_round=reflect):
    """run_round executes full trajectories and writes its bound ROUND.json.

    Service errors stop progression. Only completed unsuccessful trajectories
    become the next round's inputs; successful cases keep their prior outcomes.
    """
    if suite not in memory.SUITES:
        raise ValueError("Invalid recovery suite")
    keys=[case["logical_key"] for case in cases]
    if not keys or len(keys)!=len(set(keys)):
        raise ValueError("Recovery cases must have unique nonempty logical keys")
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    current=memory.empty(output/"M0",suite)
    pending=list(cases);completed={};rounds=[]
    remaining_config=dict(model_config or {})
    attempted,spent=0,Decimal(0)
    result={"status":"incomplete","suite":suite,"memory_mode":"fresh",
            "original_case_count":len(keys),"rounds":rounds}
    try:
        for generation in range(3):
            path=output/("R"+str(generation))
            run_round(pending,current["root"],path,generation)
            document=read_json(path/"ROUND.json")
            expected={case["logical_key"] for case in pending}
            rows=document["cases"]
            if (document["round"]!="R"+str(generation) or
                    document["memory_snapshot_sha256"]!=current["snapshot_sha256"] or
                    set(document["expected_cases"])!=expected or
                    len(rows)!=len(expected) or {r["logical_key"] for r in rows}!=expected):
                raise ValueError("Round execution did not preserve input cases and memory")
            entry={"round":document["round"],"attempted":len(rows),
                   "memory_snapshot_sha256":current["snapshot_sha256"],"directory":path.name}
            rounds.append(entry)
            errors=[r["logical_key"] for r in rows if r["status"]!="completed"]
            if errors or document["status"]!="closed":
                entry.update(status="incomplete",nonterminal_cases=errors)
                result["reason"]="ROUND_SERVICE_OR_RESOURCE_ERROR"
                break
            if any(type(r.get("success")) is not bool for r in rows):
                raise ValueError("Completed trajectories require Boolean task success")
            completed.update({r["logical_key"]:r["success"] for r in rows})
            failed={r["logical_key"] for r in rows if not r["success"]}
            entry.update(status="completed",successes=sum(r["success"] for r in rows),
                         failures=len(failed),cumulative_successes=sum(completed.values()))
            pending=[case for case in pending if case["logical_key"] in failed]
            if generation==2:
                result.update(status="completed",final_successes=sum(completed.values()),
                              final_failures=len(pending))
                break
            parent_hash=current["snapshot_sha256"]
            current=reflect_round(path,current["root"],output/("reflection_"+str(generation+1)),
                suite=suite,generation=generation+1,model_config=remaining_config)
            cost=current.get("reflection_cost",{})
            if cost:
                entry["reflection_cost"]=cost
                attempted+=cost["attempts"];spent+=Decimal(cost["cost_upper_bound_usd"])
                if "max_requests" in remaining_config:
                    remaining_config["max_requests"]=max(0,remaining_config["max_requests"]-cost["attempts"])
                if "budget_usd" in remaining_config:
                    remaining_config["budget_usd"]=str(max(Decimal(0),Decimal(str(remaining_config["budget_usd"]))-Decimal(cost["cost_upper_bound_usd"])))
            current=memory.load(current["root"])
            if current["manifest"]["parent_snapshot_sha256"]!=parent_hash:
                raise ValueError("New reflection memory is not bound to the preceding round")
            if (current["manifest"]["fixture_only"] or
                    current["manifest"]["suite"]!=suite or
                    current["manifest"]["generation"]!=generation+1):
                raise ValueError("Invalid next-round memory")
            if current["manifest"]["provenance"].get("complete") is False:
                result["reason"]="REFLECTION_SERVICE_OR_RESOURCE_ERROR"
                break
    except BaseException as exc:
        result.update(error=type(exc).__name__,message=str(exc))
        raise
    finally:
        result.update(reflection_attempts=attempted,reflection_cost_upper_bound_usd=str(spent))
        write_json(output/"RESULT.json",result)
    return result
