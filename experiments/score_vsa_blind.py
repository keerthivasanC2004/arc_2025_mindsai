#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); ap.add_argument('--solutions',required=True); args=ap.parse_args()
raw=Path(args.predictions).read_text(); calc=hashlib.sha256(raw.encode()).hexdigest(); expected=Path(args.predictions+'.sha256').read_text().strip(); assert calc==expected,(calc,expected)
p=json.loads(raw); sol=json.load(open(args.solutions)); per={}; exact=0
for tid in p['selected']:
    pred=p['predictions'].get(tid,[]); gt=sol[tid]
    ok=(len(pred)==len(gt) and all(a==b for a,b in zip(pred,gt)))
    per[tid]=ok; exact+=int(ok)
print('VERIFIED_FREEZE_SHA256',calc)
print('EXACT_SCORE',exact,'/',len(p['selected']))
print('PER_TASK',json.dumps(per,sort_keys=True))
print('TRAIN_EXACT_TOTAL',sum(v['train_exact'] for v in p['diagnostics'].values()),'/',sum(v['n_train'] for v in p['diagnostics'].values()))
