"""Compare frozen BC policies on the common validation split only.

This is an offline diagnostic for v1, full-candidate FP32, and multi-weight
FP32 checkpoints. It never opens a held-out test shard and never writes model
weights or online code.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from model import CandidateModel, ModelConfig
from train_bc import batches

KINDS=("pass","invalid","single","pair","three","straight","set","three_straight","triple_pairs","bomb","straight_flush","rocket")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def cb(n):
    if n==1:return '1'
    if n<=3:return '2-3'
    if n<=8:return '4-8'
    if n<=16:return '9-16'
    if n<=32:return '17-32'
    if n<=64:return '33-64'
    if n<=128:return '65-128'
    if n<=256:return '129-256'
    if n<=512:return '257-512'
    return '513+'

def fresh():
    return {'rows':0,'correct':0,'choice_rows':0,'choice_correct':0,
            'demo_pass':0,'pred_pass':0,'false_pass':0,'missed_pass':0,
            'demo_size_ge4':0,'size_ge4_correct':0,'size_ge4_same':0,
            'demo_size':collections.Counter(),'pred_size':collections.Counter(),
            'demo_kind':collections.Counter(),'pred_kind':collections.Counter(),
            'margin':[],'choice_margin':[],'winner_margin':[],'pass_play_margin':[],'pass_play_margin_demo_pass':[],'pass_play_margin_demo_play':[],'row_mean':[],'row_span':[]}

def add(dst, scores, actions, pos, context, count):
    win=int(scores.argmax()); correct=bool(pos[win]); passmask=actions[:,126]>.5
    demo_pass=bool((passmask&pos).any()); pred_pass=bool(passmask[win])
    sizes=np.rint(actions[:,120]*10).astype(int); kinds=np.argmax(actions[:,108:120],axis=1)
    demo_sizes=sizes[pos]; demo_kinds={KINDS[int(x)] for x in kinds[pos]}
    demo_size_ge4=bool((demo_sizes>=4).any()); same_size=bool(demo_sizes.size and sizes[win] in set(demo_sizes.tolist()))
    vals=scores.astype(np.float64); bestpos=float(vals[pos].max()); bestneg=float(vals[~pos].max()) if (~pos).any() else bestpos
    play=vals[~passmask]; passscore=float(vals[passmask][0]) if passmask.any() else float('nan')
    d=dst; d['rows']+=1; d['correct']+=int(correct); d['choice_rows']+=int(count>1); d['choice_correct']+=int(correct and count>1)
    d['demo_pass']+=int(demo_pass); d['pred_pass']+=int(pred_pass); d['false_pass']+=int(pred_pass and not demo_pass); d['missed_pass']+=int(demo_pass and not pred_pass)
    d['demo_size_ge4']+=int(demo_size_ge4); d['size_ge4_correct']+=int(demo_size_ge4 and correct); d['size_ge4_same']+=int(demo_size_ge4 and same_size)
    d['demo_size'].update(str(int(x)) for x in set(demo_sizes.tolist())); d['pred_size'][str(int(sizes[win]))]+=1; d['demo_kind'].update(demo_kinds); d['pred_kind'][KINDS[int(kinds[win])]]+=1
    d['margin'].append(bestpos-bestneg)
    if count>1: d['choice_margin'].append(bestpos-bestneg)
    d['winner_margin'].append(float(np.partition(vals,-2)[-1]-np.partition(vals,-2)[-2]) if count>1 else 0.)
    if passmask.any() and play.size:
        pm=passscore-float(play.max()); d['pass_play_margin'].append(pm)
        d['pass_play_margin_demo_pass' if demo_pass else 'pass_play_margin_demo_play'].append(pm)
    d['row_mean'].append(float(vals.mean())); d['row_span'].append(float(np.ptp(vals)))
    return {'correct':correct,'demo_pass':demo_pass,'pred_pass':pred_pass,'size':int(sizes[win]),'demo_sizes':demo_sizes,'kind':KINDS[int(kinds[win])], 'demo_kinds':demo_kinds}

def finish(d):
    for k in ('margin','choice_margin','winner_margin','pass_play_margin','pass_play_margin_demo_pass','pass_play_margin_demo_play','row_mean','row_span'):
        a=np.asarray(d[k],dtype=float)
        d[k]={'count':len(a),'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),'min':float(a.min()),'max':float(a.max())} if len(a) else None
    n=d['rows']; c=d['choice_rows']
    d.update(accuracy=d['correct']/n,choice_accuracy=d['choice_correct']/max(1,c),demo_pass_rate=d['demo_pass']/n,pred_pass_rate=d['pred_pass']/n,pass_rate_error=(d['pred_pass']-d['demo_pass'])/n,false_pass_rate=d['false_pass']/max(1,n-d['demo_pass']),missed_pass_rate=d['missed_pass']/max(1,d['demo_pass']),size_ge4_accuracy=d['size_ge4_correct']/max(1,d['demo_size_ge4']),size_ge4_same_size_recall=d['size_ge4_same']/max(1,d['demo_size_ge4']))
    for k in ('demo_size','pred_size','demo_kind','pred_kind'): d[k]=dict(sorted(d[k].items()))
    return d

def eval_policy(path, files, args):
    ckpt=torch.load(path,map_location='cpu',weights_only=True); model=CandidateModel(ModelConfig(**ckpt['config'])).float().cuda().eval(); model.load_state_dict(ckpt['state_dict'],strict=True)
    all_d=fresh(); contexts={'lead':fresh(),'follow':fresh()}; bins={}; sizes={}; kinds={}
    with torch.inference_mode():
        for batch in batches(files,args.batch_size,0,False):
            positives=batch.pop('positives'); batch={k:v.cuda() for k,v in batch.items()}; scores=model(**batch).float().cpu().numpy(); acts=batch['actions'].cpu().numpy(); mask=batch['mask'].cpu().numpy(); states=batch['state'].cpu().numpy(); pos=positives.cpu().numpy()
            for i in range(len(scores)):
                n=int(mask[i].sum()); aa=acts[i,:n]; ss=scores[i,:n]; pp=pos[i,:n]; context='lead' if states[i,125]>.5 else 'follow'; outcome=add(all_d,ss,aa,pp,context,n); add(contexts[context],ss,aa,pp,context,n)
                b=cb(n); bins.setdefault(b,fresh()); add(bins[b],ss,aa,pp,context,n)
                for size in set(np.rint(aa[pp,120]*10).astype(int).tolist()): sizes.setdefault(str(int(size)),fresh()); add(sizes[str(int(size))],ss,aa,pp,context,n)
                for kind in outcome['demo_kinds']: kinds.setdefault(kind,fresh()); add(kinds[kind],ss,aa,pp,context,n)
    return {'checkpoint_sha256':sha(path),'selected_epoch':ckpt.get('provenance',{}).get('selected_epoch'),'all':finish(all_d),'context':{k:finish(v) for k,v in contexts.items()},'candidate_bin':{k:finish(v) for k,v in sorted(bins.items())},'demo_size':{k:finish(v) for k,v in sorted(sizes.items(),key=lambda x:int(x[0]))},'demo_kind':{k:finish(v) for k,v in sorted(kinds.items())}}

def md(report):
    lines=['# BC policy comparison (common validation only)','', 'Frozen v1, full-candidate FP32, and multi-weight FP32 checkpoints are scored on byte-identical validation shards. No held-out test shard is opened.','', '| policy | raw accuracy | choice accuracy | demo pass | predicted pass | pass error | false pass on play | missed pass | size >=4 exact | size >=4 same-size | margin mean / p50 | score span p50 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,p in report['policies'].items():
        d=p['all']; lines.append(f"| {name} | {d['accuracy']:.3%} | {d['choice_accuracy']:.3%} | {d['demo_pass_rate']:.3%} | {d['pred_pass_rate']:.3%} | {d['pass_rate_error']:+.3%} | {d['false_pass_rate']:.3%} | {d['missed_pass_rate']:.3%} | {d['size_ge4_accuracy']:.3%} | {d['size_ge4_same_size_recall']:.3%} | {d['margin']['mean']:.4f} / {d['margin']['p50']:.4f} | {d['row_span']['p50']:.4f} |")
    lines+=['','## Context','', '| policy | context | rows | accuracy | choice accuracy | demo pass | predicted pass | pass error | margin mean | margin p50 | score span p50 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,p in report['policies'].items():
        for c,d in p['context'].items(): lines.append(f"| {name} | {c} | {d['rows']:,} | {d['accuracy']:.2%} | {d['choice_accuracy']:.2%} | {d['demo_pass_rate']:.2%} | {d['pred_pass_rate']:.2%} | {d['pass_rate_error']:+.2%} | {d['margin']['mean']:.4f} | {d['margin']['p50']:.4f} | {d['row_span']['p50']:.4f} |")
    lines+=['','## Candidate-count bins','', '| policy | bin | rows | accuracy | choice accuracy | demo pass | predicted pass | pass error | margin mean |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    order=['1','2-3','4-8','9-16','17-32','33-64','65-128','129-256','257-512','513+']
    for name,p in report['policies'].items():
        for b in order:
            if b not in p['candidate_bin']:continue
            d=p['candidate_bin'][b]; lines.append(f"| {name} | {b} | {d['rows']:,} | {d['accuracy']:.2%} | {d['choice_accuracy']:.2%} | {d['demo_pass_rate']:.2%} | {d['pred_pass_rate']:.2%} | {d['pass_rate_error']:+.2%} | {d['margin']['mean']:.4f} |")
    lines+=['','## Demonstrated size: exact and same-size recall','', '| policy | size | rows | exact accuracy | same-size recall | predicted size distribution |','|---|---:|---:|---:|---:|---|']
    for name,p in report['policies'].items():
        for size,d in p['demo_size'].items(): lines.append(f"| {name} | {size} | {d['rows']:,} | {d['accuracy']:.2%} | {d['size_ge4_same_size_recall'] if int(size)>=4 else 'n/a'} | {d['pred_size']} |")
    lines+=['','## Demonstrated kind: exact accuracy and prediction counts','', '| policy | kind | rows | exact accuracy | predicted kind counts |','|---|---|---:|---:|---|']
    for name,p in report['policies'].items():
        for kind,d in p['demo_kind'].items(): lines.append(f"| {name} | {kind} | {d['rows']:,} | {d['accuracy']:.2%} | {d['pred_kind']} |")
    lines+=['','## Interpretation','', 'The score margin is best-positive minus best-negative. Positive margin with low exact accuracy points to concrete-action ambiguity; negative margin means the model ranks a negative action above every demonstrated action. A pass/play margin is stored in JSON for follow rows. Large size recall with poor paired-game score should be read together with pass rate, candidate-bin margins, and kind distributions; imitation gains do not establish team-level usefulness.', '']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=ROOT/'data/processed/bc-v1'); p.add_argument('--output',type=Path,default=ROOT/'reports/bc-policy-comparison.json'); p.add_argument('--markdown',type=Path,default=ROOT/'reports/bc-policy-comparison.md'); p.add_argument('--batch-size',type=int,default=64); args=p.parse_args()
    files=sorted((args.data/'validation').glob('*.npz')); 
    if not files:raise ValueError('validation shards required')
    report={'schema':'oxbot-bc-policy-comparison-v1','status':'complete','split':'validation','test_data_opened':False,'validation_shards_sha256':{x.name:sha(x) for x in files},'script_sha256':sha(__file__),'policies':{}}
    specs={'v1':ROOT/'models/bc-v1/best.pt','v2-full-fp32':ROOT/'models/bc-v2-full-fp32/best.pt','multi2':ROOT/'models/bc-v2-full-fp32-multi2/best.pt'}
    for name,path in specs.items(): report['policies'][name]=eval_policy(path,files,args); print(json.dumps({'policy':name,'all':report['policies'][name]['all']}),flush=True)
    report['validation_shards_byte_identical_v2']=all((ROOT/'data/processed/bc-v2-full'/'validation'/name).is_file() and sha(ROOT/'data/processed/bc-v2-full'/'validation'/name)==h for name,h in report['validation_shards_sha256'].items())
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); args.markdown.write_text(md(report),encoding='utf-8')

if __name__=='__main__':main()
