#!/usr/bin/env python3
import json, math, os
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from experiments.ncp_full_1000_120 import (
    TinyNCPARC, load_json, build_pretrain_pairs, collate_pairs, adapt, set_seed,
    MAX_HW, N_COLORS
)

ROOT = Path('prior')
DATA = Path('data')
SEED = 260913


def locate(name):
    hits=list(ROOT.rglob(name))
    if not hits: raise FileNotFoundError(name)
    return hits[0]


def shape(g): return (len(g), len(g[0]) if g else 0)


def score_pair(pred,gold):
    ps,gs=shape(pred),shape(gold)
    sh=ps==gs
    exact=sh and pred==gold
    corr=0; total=gs[0]*gs[1]
    if sh and total:
        corr=int((np.asarray(pred)==np.asarray(gold)).sum())
    return exact,sh,corr,total


def prediction_diagnostics(preds, challenges, solutions):
    n=shape_ok=exact=0
    cond_corr=cond_total=0
    penal_corr=gold_total=0
    input_shape_heur=0
    pred_same_input_shape=0
    gold_same_input_shape=0
    pred_copy=pred_copy_n=0
    gold_copy=gold_copy_n=0
    pred_zero=pred_cells=0
    gold_zero=gold_cells=0
    pred_shapes=Counter(); gold_shapes=Counter(); in_shapes=Counter()
    pred_colors=Counter(); gold_colors=Counter()
    for tid,task in challenges.items():
        for q,pred,gold in zip(task['test'],preds[tid],solutions[tid]):
            x=q['input']; n+=1
            ps,gs,is_=shape(pred),shape(gold),shape(x)
            pred_shapes[ps]+=1; gold_shapes[gs]+=1; in_shapes[is_]+=1
            input_shape_heur += int(is_==gs)
            pred_same_input_shape += int(ps==is_)
            gold_same_input_shape += int(gs==is_)
            ex,sh,corr,total=score_pair(pred,gold)
            exact+=int(ex); shape_ok+=int(sh); penal_corr+=corr; gold_total+=total
            if sh: cond_corr+=corr; cond_total+=total
            pa=np.asarray(pred); ga=np.asarray(gold); xa=np.asarray(x)
            pred_zero += int((pa==0).sum()); pred_cells += pa.size
            gold_zero += int((ga==0).sum()); gold_cells += ga.size
            pred_colors.update(pa.ravel().tolist()); gold_colors.update(ga.ravel().tolist())
            oh=min(pa.shape[0],xa.shape[0]); ow=min(pa.shape[1],xa.shape[1])
            if oh and ow:
                pred_copy += int((pa[:oh,:ow]==xa[:oh,:ow]).sum()); pred_copy_n += oh*ow
            oh=min(ga.shape[0],xa.shape[0]); ow=min(ga.shape[1],xa.shape[1])
            if oh and ow:
                gold_copy += int((ga[:oh,:ow]==xa[:oh,:ow]).sum()); gold_copy_n += oh*ow
    return {
        'n_test_grids':n,
        'exact':exact,
        'shape_accuracy':shape_ok/n,
        'conditional_pixel_accuracy_given_correct_shape': cond_corr/max(1,cond_total),
        'penalized_pixel_accuracy':penal_corr/max(1,gold_total),
        'input_shape_identity_heuristic_accuracy':input_shape_heur/n,
        'predicted_shape_equals_input_fraction':pred_same_input_shape/n,
        'gold_shape_equals_input_fraction':gold_same_input_shape/n,
        'pred_same_position_copy_fraction_overlap':pred_copy/max(1,pred_copy_n),
        'gold_same_position_copy_fraction_overlap':gold_copy/max(1,gold_copy_n),
        'pred_zero_fraction':pred_zero/max(1,pred_cells),
        'gold_zero_fraction':gold_zero/max(1,gold_cells),
        'top_pred_shapes':[[list(k),v] for k,v in pred_shapes.most_common(10)],
        'top_gold_shapes':[[list(k),v] for k,v in gold_shapes.most_common(10)],
        'top_input_shapes':[[list(k),v] for k,v in in_shapes.most_common(10)],
        'pred_color_freq':dict(sorted(pred_colors.items())),
        'gold_color_freq':dict(sorted(gold_colors.items())),
    }


def chunk_mismatch_stats(solutions):
    gaps=[]; active_counts=[]; span_counts=[]; examples=[]
    for tid,sols in solutions.items():
        for gold in sols:
            h,w=shape(gold)
            active=sorted({(r*MAX_HW+c)//4 for r in range(h) for c in range(w)})
            if not active: continue
            span=active[-1]-active[0]+1
            gap=span-len(active)
            gaps.append(gap); active_counts.append(len(active)); span_counts.append(span)
            if len(examples)<5 and gap>0:
                examples.append({'shape':[h,w],'active_chunks':len(active),'span_chunks':span,'inactive_gru_steps_inside_span':gap,'inference_chunks':math.ceil(h*w/4)})
    return {
        'grids':len(gaps),
        'fraction_with_training_inference_chunk_gap':sum(g>0 for g in gaps)/len(gaps),
        'mean_inactive_gru_steps_inside_active_span':float(np.mean(gaps)),
        'median_inactive_gru_steps_inside_active_span':float(np.median(gaps)),
        'mean_training_active_chunks':float(np.mean(active_counts)),
        'mean_training_span_chunks':float(np.mean(span_counts)),
        'examples':examples,
    }


def codebook_diagnostics(state, train_ch, train_sol, device='cpu'):
    m=TinyNCPARC().to(device); m.load_state_dict(state); m.eval()
    records,_=build_pretrain_pairs(train_ch,train_sol)
    # deterministic spread across the 4308 pairs
    idx=np.linspace(0,len(records)-1,num=min(512,len(records)),dtype=int)
    counters=[Counter() for _ in range(m.S)]
    cvals=[]
    with torch.no_grad():
        for s in range(0,len(idx),32):
            pairs=[records[i][1] for i in idx[s:s+32]]
            src,sm,tgt,tm,h,w=collate_pairs(pairs,device)
            spatial,g=m.encode(src,sm); base=m.base_positions(spatial,g)
            c,q,fb,ex,gv=m.concept_feedback_teacher(base,tgt,tm)
            seg=c.view(c.size(0),c.size(1),m.S,m.ds)
            cb=m.codebooks.view(1,1,m.S,m.N,m.ds)
            dist=((seg.unsqueeze(3)-cb)**2).sum(-1)
            ids=dist.argmin(-1)
            for bi in range(c.size(0)):
                valid=gv[bi]
                if valid.any():
                    cvals.append(c[bi,valid].cpu())
                    for ss in range(m.S): counters[ss].update(ids[bi,valid,ss].cpu().tolist())
    allc=torch.cat(cvals,0) if cvals else torch.empty(0,m.d)
    usage=[]
    for ss,cnt in enumerate(counters):
        total=sum(cnt.values()); probs=np.array([cnt.get(i,0)/total for i in range(m.N)]) if total else np.zeros(m.N)
        nz=probs[probs>0]; ent=float(-(nz*np.log(nz)).sum()/math.log(m.N)) if len(nz) else 0.0
        usage.append({'subspace':ss,'unique_codes_used':sum(v>0 for v in cnt.values()),'normalized_entropy':ent,'top_codes':cnt.most_common(5)})
    return {
        'sampled_pairs':len(idx),
        'concept_vector_variance_mean':float(allc.var(0,unbiased=False).mean()) if allc.numel() else 0.0,
        'codebook_usage':usage,
        'copy_scale':float(state['copy_scale']),
        'feedback_scale':float(state['feedback_scale']),
    }


def support_and_feedback_diagnostics(control_state,ncp_state,eval_ch,eval_sol,device='cpu'):
    tids=sorted(eval_ch)[::3]  # 40 of 120, deterministic
    res={
        'sampled_tasks':len(tids),
        'control':{'support_exact':0,'support_grids':0,'support_shape':0,'support_pixel_corr':0,'support_pixels':0},
        'ncp':{'support_exact':0,'support_grids':0,'support_shape':0,'support_pixel_corr':0,'support_pixels':0},
        'ncp_feedback_on_test':{'exact':0,'shape':0,'pixel_corr':0,'pixels':0,'grids':0},
        'ncp_feedback_off_test':{'exact':0,'shape':0,'pixel_corr':0,'pixels':0,'grids':0},
    }
    for ti,tid in enumerate(tids,1):
        task=eval_ch[tid]; support=[{'input':p['input'],'output':p['output']} for p in task['train']]
        cm=adapt(control_state,support,False,device,12,SEED+1000+(sorted(eval_ch).index(tid)+1)*17)
        nm=adapt(ncp_state,support,True,device,12,SEED+1000+(sorted(eval_ch).index(tid)+1)*17)
        for arm,m,use in [('control',cm,False),('ncp',nm,True)]:
            task_all=True
            for p in support:
                pred=m.predict(p['input'],use)
                ex,sh,corr,total=score_pair(pred,p['output'])
                task_all &= ex
                r=res[arm]; r['support_grids']+=1; r['support_shape']+=int(sh); r['support_pixel_corr']+=corr; r['support_pixels']+=total
            res[arm]['support_exact']+=int(task_all)
        for q,gold in zip(task['test'],eval_sol[tid]):
            for label,use in [('ncp_feedback_on_test',True),('ncp_feedback_off_test',False)]:
                pred=nm.predict(q['input'],use)
                ex,sh,corr,total=score_pair(pred,gold)
                r=res[label]; r['grids']+=1; r['exact']+=int(ex); r['shape']+=int(sh); r['pixel_corr']+=corr; r['pixels']+=total
        if ti%10==0: print(json.dumps({'support_diag_tasks_done':ti}),flush=True)
    for arm in ['control','ncp']:
        r=res[arm]
        r['support_task_exact_rate']=r['support_exact']/len(tids)
        r['support_shape_accuracy']=r['support_shape']/max(1,r['support_grids'])
        r['support_penalized_pixel_accuracy']=r['support_pixel_corr']/max(1,r['support_pixels'])
    for label in ['ncp_feedback_on_test','ncp_feedback_off_test']:
        r=res[label]
        r['exact_rate']=r['exact']/max(1,r['grids']); r['shape_accuracy']=r['shape']/max(1,r['grids']); r['penalized_pixel_accuracy']=r['pixel_corr']/max(1,r['pixels'])
    return res


def main():
    torch.set_num_threads(min(4,os.cpu_count() or 2)); set_seed(SEED)
    eval_ch=load_json(DATA/'arc-agi-2_evaluation_challenges.json'); eval_sol=load_json(DATA/'arc-agi-2_evaluation_solutions.json')
    train_ch=load_json(DATA/'arc-agi-2_training_challenges.json'); train_sol=load_json(DATA/'arc-agi-2_training_solutions.json')
    cp=load_json(locate('control_predictions.json')); npred=load_json(locate('ncp_predictions.json'))
    cstate=torch.load(locate('control_pretrained.pt'),map_location='cpu'); nstate=torch.load(locate('ncp_pretrained.pt'),map_location='cpu')
    out={
        'control_predictions':prediction_diagnostics(cp,eval_ch,eval_sol),
        'ncp_predictions':prediction_diagnostics(npred,eval_ch,eval_sol),
        'chunk_sequence_mismatch':chunk_mismatch_stats(eval_sol),
        'control_codebook':codebook_diagnostics(cstate,train_ch,train_sol),
        'ncp_codebook':codebook_diagnostics(nstate,train_ch,train_sol),
        'support_and_feedback':support_and_feedback_diagnostics(cstate,nstate,eval_ch,eval_sol),
    }
    Path('diagnostics').mkdir(exist_ok=True)
    Path('diagnostics/diagnostics.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__': main()
