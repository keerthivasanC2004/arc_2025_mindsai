#!/usr/bin/env python3
import argparse, hashlib, json, os, sys, time, types
from pathlib import Path

import interruptingcow

MAX_SOLVING_TIME = 1000

class SolvingTimeoutException(Exception):
    pass

EXCLUDE = {
    "80a900e0","45a5af55","4c7dc4dd","31f7f899","7b5033c1",
    "67e490f4","269e22fb","4e34c42c","981571dc","e8686506",
    "0934a4d8","eee78d87","36a08778","38007db0","7491f3cf",
    "5545f144","88bcf3b4","4c3d4a41","78332cb0","e12f9a14",
}


def task_features(task):
    vals=[]
    for p in task["train"]:
        a,b=p["input"],p["output"]
        vals += [len(a),len(a[0]),len(b),len(b[0]),len({x for r in a for x in r}),len({x for r in b for x in r})]
    while len(vals)<36: vals.append(0)
    vals=vals[:36]
    n=(sum(x*x for x in vals)**0.5) or 1
    return [x/n for x in vals]


def sim(a,b):
    d=sum((x-y)**2 for x,y in zip(a,b))
    return pow(2.718281828,-d/0.02)


def select(tasks,n=10,threshold=.10):
    ids=sorted(k for k in tasks if k not in EXCLUDE)
    feats={k:task_features(tasks[k]) for k in ids}
    chosen=[ids[0]]
    while len(chosen)<n:
        candidates=[]
        for k in ids:
            if k in chosen: continue
            mx=max(sim(feats[k],feats[c]) for c in chosen)
            candidates.append((mx,k))
        candidates.sort()
        mx,k=candidates[0]
        if mx>=threshold:
            raise RuntimeError(f"Cannot select {n} tasks below similarity threshold; next similarity={mx}")
        chosen.append(k)
    return chosen,max(sim(feats[a],feats[b]) for i,a in enumerate(chosen) for b in chosen[i+1:])


def dummy_task(task):
    t={"train":task["train"],"test":[]}
    for q in task["test"]:
        inp=q["input"]
        t["test"].append({"input":inp,"output":[[0 for _ in row] for row in inp]})
    return t


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--challenges",required=True)
    ap.add_argument("--arcvsa",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--extra-exclude-file")
    args=ap.parse_args()

    if args.extra_exclude_file and Path(args.extra_exclude_file).exists():
        extra={x.strip() for x in Path(args.extra_exclude_file).read_text().splitlines() if x.strip()}
        EXCLUDE.update(extra)
        print("EXTRA_EXCLUDED",sorted(extra),flush=True)

    tasks=json.load(open(args.challenges))
    selected,mx=select(tasks)
    if set(selected) & EXCLUDE:
        raise RuntimeError("Protected/spent task selected")

    print("SELECTED",selected,flush=True)
    print("MAX_PAIRWISE_SIMILARITY",mx,flush=True)
    print("MAX_SOLVING_TIME_SECONDS",MAX_SOLVING_TIME,flush=True)

    sys.path.insert(0,str(Path(args.arcvsa)/"src"))
    from objobj_solver import ObjObjSolver
    predictions={}
    diagnostics={}

    for index,tid in enumerate(selected,1):
        task=dummy_task(tasks[tid])
        captured={}
        started=time.monotonic()
        print(f"TASK_START {index}/10 {tid}",flush=True)

        try:
            with interruptingcow.timeout(MAX_SOLVING_TIME, exception=SolvingTimeoutException):
                solver=ObjObjSolver(task)

                def blind_print_results(self, train_gen_out_grids, test_gen_out_grids):
                    train_correct=sum(int(g==y) for g,y in zip(train_gen_out_grids,self.train_out_grids))
                    captured["pred"]=[g.get_data().astype(int).tolist() for g in test_gen_out_grids]
                    captured["train_exact"]=train_correct
                    return train_correct/len(self.train_out_grids),0.0

                solver.print_results=types.MethodType(blind_print_results,solver)
                solver.solve_task()

            elapsed=time.monotonic()-started
            predictions[tid]=captured.get("pred",[])
            diagnostics[tid]={
                "train_exact":captured.get("train_exact",0),
                "n_train":len(task["train"]),
                "error":None,
                "elapsed_seconds":round(elapsed,3),
            }
            print(f"TASK_DONE {index}/10 {tid} elapsed={elapsed:.3f}s train_exact={captured.get('train_exact',0)}/{len(task['train'])}",flush=True)

        except SolvingTimeoutException:
            elapsed=time.monotonic()-started
            predictions[tid]=[]
            diagnostics[tid]={
                "train_exact":0,
                "n_train":len(task["train"]),
                "error":f"SolvingTimeoutException:timed out after {MAX_SOLVING_TIME} seconds",
                "elapsed_seconds":round(elapsed,3),
            }
            print(f"TASK_TIMEOUT {index}/10 {tid} elapsed={elapsed:.3f}s",flush=True)

        except Exception as e:
            elapsed=time.monotonic()-started
            predictions[tid]=[]
            diagnostics[tid]={
                "train_exact":0,
                "n_train":len(task["train"]),
                "error":type(e).__name__+":"+str(e),
                "elapsed_seconds":round(elapsed,3),
            }
            print(f"TASK_ERROR {index}/10 {tid} elapsed={elapsed:.3f}s error={type(e).__name__}:{e}",flush=True)

    payload={
        "selected":selected,
        "max_pairwise_similarity":mx,
        "max_solving_time_seconds":MAX_SOLVING_TIME,
        "predictions":predictions,
        "diagnostics":diagnostics,
        "upstream":"ijoffe/ARC-VSA-2025@c031a9c6b4885ab03b28fbfdcd97b6b3693df564",
    }
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"))
    Path(args.out).write_text(raw)
    sha=hashlib.sha256(raw.encode()).hexdigest()
    Path(args.out+".sha256").write_text(sha+"\n")
    print("DIAGNOSTICS",json.dumps(diagnostics,sort_keys=True),flush=True)
    print("FROZEN_SHA256",sha,flush=True)

if __name__=="__main__": main()
