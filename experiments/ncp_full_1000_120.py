#!/usr/bin/env python3
import argparse, copy, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_HW = 30
N_COLORS = 10
PAD_COLOR = 10


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def d4_grid(grid, op:int):
    a=np.asarray(grid,dtype=np.int64)
    if op==0: b=a
    elif op==1: b=np.rot90(a,1)
    elif op==2: b=np.rot90(a,2)
    elif op==3: b=np.rot90(a,3)
    elif op==4: b=np.fliplr(a)
    elif op==5: b=np.flipud(a)
    elif op==6: b=a.T
    elif op==7: b=np.rot90(a.T,2)
    else: raise ValueError(op)
    return b.copy().tolist()


def transform_pair(pair,op):
    return {'input':d4_grid(pair['input'],op),'output':d4_grid(pair['output'],op)}


def load_json(path):
    with open(path,'r') as f: return json.load(f)


def build_pretrain_pairs(challenges, solutions):
    out=[]; task_counts={}
    for tid,task in challenges.items():
        n0=len(out)
        for p in task['train']:
            out.append((tid, {'input':p['input'],'output':p['output']}))
        sols=solutions.get(tid,[])
        for q,gold in zip(task.get('test',[]),sols):
            out.append((tid, {'input':q['input'],'output':gold}))
        task_counts[tid]=len(out)-n0
    return out,task_counts


def collate_pairs(pairs,device):
    B=len(pairs)
    src=torch.full((B,MAX_HW,MAX_HW),PAD_COLOR,dtype=torch.long,device=device)
    tgt=torch.zeros((B,MAX_HW,MAX_HW),dtype=torch.long,device=device)
    src_mask=torch.zeros((B,MAX_HW,MAX_HW),dtype=torch.bool,device=device)
    tgt_mask=torch.zeros((B,MAX_HW,MAX_HW),dtype=torch.bool,device=device)
    hs=[]; ws=[]
    for b,p in enumerate(pairs):
        x=np.asarray(p['input'],dtype=np.int64); y=np.asarray(p['output'],dtype=np.int64)
        hi,wi=x.shape; ho,wo=y.shape
        if max(hi,wi,ho,wo)>MAX_HW: raise ValueError('ARC grid exceeds 30x30')
        src[b,:hi,:wi]=torch.from_numpy(x).to(device)
        tgt[b,:ho,:wo]=torch.from_numpy(y).to(device)
        src_mask[b,:hi,:wi]=True; tgt_mask[b,:ho,:wo]=True
        hs.append(ho-1); ws.append(wo-1)
    return src,src_mask,tgt,tgt_mask,torch.tensor(hs,device=device),torch.tensor(ws,device=device)


class TinyNCPARC(nn.Module):
    def __init__(self,d=32,chunk=4,S=4,N=16):
        super().__init__(); assert d%S==0
        self.d=d; self.chunk=chunk; self.S=S; self.N=N; self.ds=d//S
        self.in_emb=nn.Embedding(11,d)
        self.out_emb=nn.Embedding(10,d)
        self.row=nn.Embedding(MAX_HW,d); self.col=nn.Embedding(MAX_HW,d)
        self.conv1=nn.Conv2d(d+1,d,3,padding=1)
        self.conv2=nn.Conv2d(d,d,3,padding=1)
        self.gproj=nn.Linear(d,d)
        self.base_norm=nn.LayerNorm(d)
        self.shape=nn.Sequential(nn.Linear(d,2*d),nn.ReLU(),nn.Linear(2*d,2*MAX_HW))
        self.token=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,N_COLORS))
        self.copy_scale=nn.Parameter(torch.tensor(1.2))
        self.codebooks=nn.Parameter(torch.randn(S,N,self.ds)*0.12)
        self.cgru=nn.GRU(d,d,batch_first=True)
        self.chead=nn.Linear(d,S*N)
        self.feedback_scale=nn.Parameter(torch.tensor(0.5))

    def encode(self,src,src_mask):
        x=self.in_emb(src).permute(0,3,1,2)
        m=src_mask.float().unsqueeze(1)
        h=torch.cat([x*m,m],1)
        h=F.gelu(self.conv1(h)); h=F.gelu(self.conv2(h)+h[:,:self.d])
        den=m.sum((2,3)).clamp_min(1.)
        g=(h*m).sum((2,3))/den
        return h.permute(0,2,3,1),g

    def base_positions(self,spatial,g):
        B=spatial.size(0); device=spatial.device
        rr=torch.arange(MAX_HW,device=device).view(1,MAX_HW,1).expand(B,MAX_HW,MAX_HW)
        cc=torch.arange(MAX_HW,device=device).view(1,1,MAX_HW).expand(B,MAX_HW,MAX_HW)
        base=spatial+self.row(rr)+self.col(cc)+self.gproj(g)[:,None,None,:]
        return self.base_norm(base).reshape(B,MAX_HW*MAX_HW,self.d)

    def quantize(self,c):
        B,G,D=c.shape; seg=c.view(B,G,self.S,self.ds)
        cb=self.codebooks.view(1,1,self.S,self.N,self.ds)
        dist=((seg.unsqueeze(3)-cb)**2).sum(-1)
        idx=dist.argmin(-1)
        q=torch.cat([self.codebooks[s][idx[:,:,s]] for s in range(self.S)],-1)
        return q

    def concept_feedback_teacher(self,base,tgt,tgt_mask):
        B,L,D=base.shape; k=self.chunk; G=(L+k-1)//k
        te=self.out_emb(tgt.reshape(B,L)); z=base+te; mask=tgt_mask.reshape(B,L)
        if G*k>L:
            pad=G*k-L; z=F.pad(z,(0,0,0,pad)); mask=F.pad(mask,(0,pad),value=False)
        zg=z.view(B,G,k,D); mg=mask.view(B,G,k)
        den=mg.sum(-1,keepdim=True).clamp_min(1)
        c=(zg*mg.unsqueeze(-1)).sum(2)/den; gv=mg.any(-1); q=self.quantize(c)
        h,_=self.cgru(c); logits=self.chead(h).view(B,G,self.S,self.N); fb=torch.zeros_like(c)
        if G>1:
            p=logits[:,:-1].softmax(-1)
            parts=[torch.einsum('bgn,nd->bgd',p[:,:,s],self.codebooks[s]) for s in range(self.S)]
            fb[:,1:]=torch.cat(parts,-1)
        ex=fb.repeat_interleave(k,1)[:,:L]
        return c,q,fb,ex,gv

    def copy_logits(self,src):
        flat=src.reshape(src.size(0),-1); ok=flat<N_COLORS; safe=flat.clamp_max(N_COLORS-1)
        cp=F.one_hot(safe,num_classes=N_COLORS).float()*ok.unsqueeze(-1)
        return self.copy_scale*cp

    def forward(self,src,src_mask,tgt,tgt_mask,hcls,wcls,use_ncp):
        spatial,g=self.encode(src,src_mask); base=self.base_positions(spatial,g)
        sh=self.shape(g); hlog=sh[:,:MAX_HW]; wlog=sh[:,MAX_HW:]
        c,q,fb,ex,gv=self.concept_feedback_teacher(base,tgt,tgt_mask)
        z=base + (self.feedback_scale*ex if use_ncp else 0.0)
        logits=self.token(z)+self.copy_logits(src)
        flat_t=tgt.reshape(tgt.size(0),-1); flat_m=tgt_mask.reshape(tgt.size(0),-1)
        ce=F.cross_entropy(logits.reshape(-1,N_COLORS),flat_t.reshape(-1),reduction='none').view_as(flat_t)
        tok=(ce*flat_m).sum()/flat_m.sum().clamp_min(1)
        shape_loss=0.5*(F.cross_entropy(hlog,hcls)+F.cross_entropy(wlog,wcls))
        vqm=gv.unsqueeze(-1); vq=((c.detach()-q)**2*vqm).sum()/(vqm.sum()*self.d).clamp_min(1)
        ncp=torch.tensor(0.,device=src.device)
        if c.size(1)>1:
            m=gv[:,1:].unsqueeze(-1)
            ncp=(((fb[:,1:]-c[:,1:].detach())**2)*m).sum()/((m.sum()*self.d).clamp_min(1))
        loss=tok+0.35*shape_loss
        if use_ncp: loss=loss+0.25*ncp+0.10*vq
        return loss,{'tok':float(tok.detach()),'shape':float(shape_loss.detach()),'ncp':float(ncp.detach()),'vq':float(vq.detach())}

    @torch.no_grad()
    def predict(self,grid,use_ncp):
        self.eval(); device=next(self.parameters()).device; x=np.asarray(grid,dtype=np.int64); hi,wi=x.shape
        src=torch.full((1,MAX_HW,MAX_HW),PAD_COLOR,dtype=torch.long,device=device)
        sm=torch.zeros((1,MAX_HW,MAX_HW),dtype=torch.bool,device=device)
        src[0,:hi,:wi]=torch.from_numpy(x).to(device); sm[0,:hi,:wi]=True
        spatial,g=self.encode(src,sm); base=self.base_positions(spatial,g)
        sh=self.shape(g); ho=int(sh[:,:MAX_HW].argmax(-1).item()+1); wo=int(sh[:,MAX_HW:].argmax(-1).item()+1)
        L=ho*wo; k=self.chunk; positions=[r*MAX_HW+c for r in range(ho) for c in range(wo)]
        out=[]; hidden=None; next_fb=torch.zeros((1,self.d),device=device); copy=self.copy_logits(src)[0]
        for a in range(0,L,k):
            ids=positions[a:min(a+k,L)]; b=base[0,ids]; z=b+(self.feedback_scale*next_fb if use_ncp else 0.0)
            lg=self.token(z)+copy[ids]; pred=lg.argmax(-1); out.extend(pred.tolist())
            c=(b+self.out_emb(pred)).mean(0,keepdim=True).unsqueeze(1); h,hidden=self.cgru(c,hidden)
            qlog=self.chead(h[:,-1]).view(1,self.S,self.N)
            parts=[torch.einsum('bn,nd->bd',qlog[:,s].softmax(-1),self.codebooks[s]) for s in range(self.S)]
            next_fb=torch.cat(parts,-1)
        return np.asarray(out,dtype=np.int64).reshape(ho,wo).tolist()


def param_count(m): return sum(p.numel() for p in m.parameters())


def make_schedule(n,epochs,batch_size,seed):
    rng=random.Random(seed); schedule=[]
    for ep in range(epochs):
        idx=list(range(n)); rng.shuffle(idx); ops=[rng.randrange(8) for _ in idx]
        for s in range(0,n,batch_size): schedule.append((idx[s:s+batch_size],ops[s:s+batch_size]))
    return schedule


def train_pretrain(model,records,schedule,use_ncp,device,lr=2e-3):
    model.train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(1,len(schedule)),eta_min=lr*0.1)
    t0=time.time(); last={}
    for step,(idxs,ops) in enumerate(schedule,1):
        pairs=[transform_pair(records[i][1],op) for i,op in zip(idxs,ops)]
        batch=collate_pairs(pairs,device); opt.zero_grad(set_to_none=True); loss,parts=model(*batch,use_ncp=use_ncp)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step(); last=parts
        if step%100==0 or step==len(schedule):
            print(json.dumps({'phase':'pretrain','arm':'ncp' if use_ncp else 'control','step':step,'steps':len(schedule),'loss':float(loss.detach()),**parts}),flush=True)
    return {'seconds':time.time()-t0,'last_loss':float(loss.detach()),**last}


def adapt(base_state,support,use_ncp,device,steps,seed,lr=2e-3):
    set_seed(seed); m=TinyNCPARC().to(device); m.load_state_dict(base_state); m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4); pool=[]
    for p in support:
        for op in range(8): pool.append(transform_pair(p,op))
    if not pool: return m
    rng=random.Random(seed+991)
    for st in range(steps):
        batch_pairs=pool if len(pool)<=24 else [pool[i] for i in rng.sample(range(len(pool)),24)]
        batch=collate_pairs(batch_pairs,device); opt.zero_grad(set_to_none=True); loss,_=m(*batch,use_ncp=use_ncp)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m


def score_predictions(preds,solutions):
    task_exact=grid_exact=grid_total=shape_exact=correct_cells=gold_cells=0; per_task={}
    for tid,plist in preds.items():
        sols=solutions[tid]; oks=[]
        for pred,gold in zip(plist,sols):
            grid_total+=1; ps=(len(pred),len(pred[0]) if pred else 0); gs=(len(gold),len(gold[0])); shape_ok=ps==gs
            shape_exact+=int(shape_ok); exact=shape_ok and pred==gold; grid_exact+=int(exact); oks.append(exact); gold_cells+=gs[0]*gs[1]
            if shape_ok: correct_cells+=int((np.asarray(pred)==np.asarray(gold)).sum())
        te=all(oks) if oks else False; task_exact+=int(te); per_task[tid]={'exact':te,'grid_exact':sum(oks),'n_grids':len(oks)}
    return {'task_exact':task_exact,'task_total':len(preds),'task_accuracy':task_exact/max(1,len(preds)),'grid_exact':grid_exact,'grid_total':grid_total,'grid_accuracy':grid_exact/max(1,grid_total),'shape_exact':shape_exact,'shape_accuracy':shape_exact/max(1,grid_total),'pixel_accuracy_penalized':correct_cells/max(1,gold_cells),'per_task':per_task}


def evaluate_arm(base_state,eval_challenges,use_ncp,device,adapt_steps,seed):
    preds={}; t0=time.time(); tids=sorted(eval_challenges)
    for ti,tid in enumerate(tids,1):
        task=eval_challenges[tid]; support=[{'input':p['input'],'output':p['output']} for p in task['train']]
        m=adapt(base_state,support,use_ncp,device,adapt_steps,seed+ti*17)
        preds[tid]=[m.predict(q['input'],use_ncp) for q in task['test']]
        if ti%20==0 or ti==len(tids): print(json.dumps({'phase':'eval_predict','arm':'ncp' if use_ncp else 'control','tasks_done':ti,'tasks_total':len(tids)}),flush=True)
    return preds,time.time()-t0


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='data'); ap.add_argument('--out-dir',default='ncp_full_results'); ap.add_argument('--seed',type=int,default=260913); ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--adapt-steps',type=int,default=12); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    torch.set_num_threads(min(4,os.cpu_count() or 2)); set_seed(args.seed); device='cpu'; d=Path(args.data_dir); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    train_ch=load_json(d/'arc-agi-2_training_challenges.json'); train_sol=load_json(d/'arc-agi-2_training_solutions.json'); eval_ch=load_json(d/'arc-agi-2_evaluation_challenges.json')
    # Evaluation solutions intentionally not loaded until predictions are frozen.
    if args.smoke:
        train_ch=dict(list(train_ch.items())[:12]); train_sol={k:train_sol[k] for k in train_ch}; eval_ch=dict(list(eval_ch.items())[:3]); args.epochs=1; args.adapt_steps=2
    records,task_counts=build_pretrain_pairs(train_ch,train_sol)
    coverage={'training_tasks':len(train_ch),'training_tasks_with_examples':sum(v>0 for v in task_counts.values()),'training_supervised_pairs':len(records),'evaluation_tasks':len(eval_ch),'max_hw':MAX_HW}; print(json.dumps({'coverage':coverage}),flush=True)
    schedule=make_schedule(len(records),args.epochs,args.batch_size,args.seed+5)
    set_seed(args.seed); init=TinyNCPARC(); init_state=copy.deepcopy(init.state_dict()); params=param_count(init)
    control=TinyNCPARC().to(device); control.load_state_dict(init_state); ncp=TinyNCPARC().to(device); ncp.load_state_dict(init_state)
    ctrain=train_pretrain(control,records,schedule,False,device); ntrain=train_pretrain(ncp,records,schedule,True,device)
    cstate=copy.deepcopy(control.state_dict()); nstate=copy.deepcopy(ncp.state_dict()); torch.save(cstate,out/'control_pretrained.pt'); torch.save(nstate,out/'ncp_pretrained.pt')
    cpreds,csec=evaluate_arm(cstate,eval_ch,False,device,args.adapt_steps,args.seed+1000); npreds,nsec=evaluate_arm(nstate,eval_ch,True,device,args.adapt_steps,args.seed+1000)
    with open(out/'control_predictions.json','w') as f: json.dump(cpreds,f)
    with open(out/'ncp_predictions.json','w') as f: json.dump(npreds,f)
    eval_sol=load_json(d/'arc-agi-2_evaluation_solutions.json'); assert set(eval_ch)==set(eval_sol)
    cs=score_predictions(cpreds,eval_sol); ns=score_predictions(npreds,eval_sol)
    summary={'protocol':{'seed':args.seed,'epochs':args.epochs,'batch_size':args.batch_size,'adapt_steps':args.adapt_steps,'evaluation_labels_loaded_after_prediction':True,'primary_metric':'exact task accuracy (all test grids exact)','note':'public evaluation is not pristine blind here because some eval labels were inspected in earlier development'},'coverage':coverage,'params_each':params,'pretrain_steps_each':len(schedule),'control_train':ctrain,'ncp_train':ntrain,'control_eval_seconds':csec,'ncp_eval_seconds':nsec,'control':cs,'ncp':ns,'delta_task_accuracy':ns['task_accuracy']-cs['task_accuracy'],'delta_grid_accuracy':ns['grid_accuracy']-cs['grid_accuracy'],'delta_shape_accuracy':ns['shape_accuracy']-cs['shape_accuracy'],'delta_pixel_accuracy_penalized':ns['pixel_accuracy_penalized']-cs['pixel_accuracy_penalized']}
    with open(out/'summary.json','w') as f: json.dump(summary,f,indent=2)
    report=f'''# NCP full ARC-AGI-2 1000/120 test\n\n- Training tasks: **{coverage['training_tasks']}**\n- Training supervised pairs: **{coverage['training_supervised_pairs']}**\n- Evaluation tasks: **{coverage['evaluation_tasks']}**\n- Parameters per arm: **{params:,}**\n- Pretraining steps per arm: **{len(schedule)}**\n- Per-evaluation-task adaptation: **{args.adapt_steps}** support-only steps\n- Evaluation labels loaded only after predictions were written: **yes**\n\n| Metric | Control | NCP | Delta |\n|---|---:|---:|---:|\n| Exact tasks | {cs['task_exact']}/{cs['task_total']} | {ns['task_exact']}/{ns['task_total']} | {100*(ns['task_accuracy']-cs['task_accuracy']):+.2f} pp |\n| Exact test grids | {cs['grid_exact']}/{cs['grid_total']} | {ns['grid_exact']}/{ns['grid_total']} | {100*(ns['grid_accuracy']-cs['grid_accuracy']):+.2f} pp |\n| Output-shape accuracy | {100*cs['shape_accuracy']:.2f}% | {100*ns['shape_accuracy']:.2f}% | {100*(ns['shape_accuracy']-cs['shape_accuracy']):+.2f} pp |\n| Penalized pixel accuracy | {100*cs['pixel_accuracy_penalized']:.2f}% | {100*ns['pixel_accuracy_penalized']:.2f}% | {100*(ns['pixel_accuracy_penalized']-cs['pixel_accuracy_penalized']):+.2f} pp |\n\n**Interpretation caution:** this is the public 120-task evaluation set, not a pristine blind test in this conversation, because a few public evaluation labels were inspected during earlier prototype development. Exact-match remains the primary endpoint.\n'''; (out/'REPORT.md').write_text(report)
    print(json.dumps({'FINAL':{k:v for k,v in summary.items() if k in ['coverage','params_each','pretrain_steps_each','control','ncp','delta_task_accuracy','delta_grid_accuracy','delta_shape_accuracy','delta_pixel_accuracy_penalized']}},default=str),flush=True)

if __name__=='__main__': main()
