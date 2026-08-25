#!/usr/bin/env python3
import argparse, hashlib, json
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--solutions',required=True);a=p.parse_args()
with open(a.predictions,'rb') as f: raw=f.read()
obj=json.loads(raw)
core={k:v for k,v in obj.items() if k!='freeze_sha256'}
calc=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert calc==obj['freeze_sha256'], 'freeze hash mismatch'
with open(a.solutions) as f: sol=json.load(f)
per={}; exact=0
for tid in obj['selected']:
    ok=obj['predictions'][tid]==sol[tid]
    per[tid]=ok; exact+=int(ok)
print('VERIFIED_FREEZE_SHA256',calc)
print('EXACT_SCORE',exact,'/',len(obj['selected']))
print('PER_TASK',json.dumps(per,sort_keys=True))
print('TRAIN_EXACT_TENSOR',sum(bool(obj['meta'][t]['tensor']['exact']) for t in obj['selected']),'/',len(obj['selected']))
print('TRAIN_EXACT_GREEDY',sum(bool(obj['meta'][t]['greedy_train_exact']) for t in obj['selected']),'/',len(obj['selected']))
