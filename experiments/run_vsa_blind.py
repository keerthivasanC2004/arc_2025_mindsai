#!/usr/bin/env python3
import argparse, hashlib, json, os, sys, types
from pathlib import Path

EXCLUDE = {"80a900e0","45a5af55","4c7dc4dd","31f7f899","7b5033c1","67e490f4","269e22fb","4e34c42c","981571dc","e8686506"}


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
    # deterministic farthest-point selection from lexicographically first task
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
    args=ap.parse_args()
    tasks=json.load(open(args.challenges))
    selected,mx=select(tasks)
    sys.path.insert(0,str(Path(args.arcvsa)/"src"))
    from objobj_solver import ObjObjSolver
    predictions={}
    diagnostics={}
    for tid in selected:
        task=dummy_task(tasks[tid])
        solver=ObjObjSolver(task)
        captured={}
        def blind_print_results(self, train_gen_out_grids, test_gen_out_grids):
            train_correct=sum(int(g==y) for g,y in zip(train_gen_out_grids,self.train_out_grids))
            captured["pred"]=[g.get_data().astype(int).tolist() for g in test_gen_out_grids]
            captured["train_exact"]=train_correct
            return train_correct/len(self.train_out_grids),0.0
        solver.print_results=types.MethodType(blind_print_results,solver)
        try:
            solver.solve_task()
            predictions[tid]=captured.get("pred",[])
            diagnostics[tid]={"train_exact":captured.get("train_exact",0),"n_train":len(task["train"]),"error":None}
        except Exception as e:
            predictions[tid]=[]
            diagnostics[tid]={"train_exact":0,"n_train":len(task["train"]),"error":type(e).__name__+":"+str(e)}
    payload={"selected":selected,"max_pairwise_similarity":mx,"predictions":predictions,"diagnostics":diagnostics,"upstream":"ijoffe/ARC-VSA-2025@c031a9c6b4885ab03b28fbfdcd97b6b3693df564"}
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"))
    Path(args.out).write_text(raw)
    sha=hashlib.sha256(raw.encode()).hexdigest()
    Path(args.out+".sha256").write_text(sha+"\n")
    print("SELECTED",selected)
    print("MAX_PAIRWISE_SIMILARITY",mx)
    print("DIAGNOSTICS",json.dumps(diagnostics,sort_keys=True))
    print("FROZEN_SHA256",sha)

if __name__=="__main__": main()
