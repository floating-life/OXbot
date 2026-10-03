"""Read-only frozen-model FP32/BF16 diagnostics on validation only."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
from model import CandidateModel, ModelConfig
from train_bc import batches, marginal_loss


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda: f.read(1 << 20), b''): h.update(x)
    return h.hexdigest()


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(a), "min": float(a.min()), "mean": float(a.mean()),
            "p50": float(np.quantile(a,.5)), "p95": float(np.quantile(a,.95)),
            "p99": float(np.quantile(a,.99)), "max": float(a.max())}


def run(checkpoint, files, args):
    original_sha = sha(checkpoint)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = CandidateModel(ModelConfig(**ckpt['config'])).float().cuda().eval()
    model.load_state_dict(ckpt['state_dict'], strict=True)
    modes = ['fp32', 'bf16', 'bf16_head2_fp32', 'fp32_rounded_bf16', 'fp32_centered_rounded_bf16']
    totals = {k: {'rows':0, 'correct':0, 'choice_rows':0, 'choice_correct':0,
                  'nll_sum':0., 'argmax_disagreement':0, 'choice_argmax_disagreement':0,
                  'choice_top_tie_rows':0, 'score_min':float('inf'), 'score_max':float('-inf')} for k in modes}
    distributions = {k: {n:[] for n in ('row_mean','row_span','row_absmax','centered_abs_error_max','raw_abs_error_max','top2_margin','unique_fraction')} for k in modes}
    examples = []
    gradient_samples = []
    # Probe last-layer FP32 on the diagnostic copy only. No source/weights edits.
    def fp32_head_hook(module, inputs, output):
        with torch.autocast(device_type='cuda', enabled=False):
            return torch.nn.functional.linear(inputs[0].float(), module.weight.float(), module.bias.float())

    with torch.inference_mode():
        for batch in batches(files, args.batch_size, 0, False):
            batch = {k:v.cuda() for k,v in batch.items()}
            positives = batch.pop('positives')
            mask = batch['mask']; counts = mask.sum(-1)
            if sum(len(x[1]) for x in gradient_samples) < args.gradient_rows:
                gradient_samples.append(({k:v.detach().cpu() for k,v in batch.items()}, positives.detach().cpu()))
            fp = model(**batch).float()
            with torch.autocast('cuda', dtype=torch.bfloat16): bf = model(**batch).float()
            hook = model.head2.register_forward_hook(fp32_head_hook)
            with torch.autocast('cuda', dtype=torch.bfloat16): last_fp = model(**batch).float()
            hook.remove()
            rounded = fp.to(torch.bfloat16).float()
            means = fp.masked_fill(~mask,0).sum(-1) / counts
            centered = (fp - means[:,None]).to(torch.bfloat16).float()
            results = {'fp32':fp,'bf16':bf,'bf16_head2_fp32':last_fp,
                       'fp32_rounded_bf16':rounded,'fp32_centered_rounded_bf16':centered}
            fp_winner = fp.argmax(-1)
            for mode,scores in results.items():
                loss = marginal_loss(scores, positives, 'none')
                winner = scores.argmax(-1)
                correct = positives.gather(1,winner[:,None]).squeeze(1)
                choice = counts > 1
                t = totals[mode]; t['rows'] += len(scores); t['correct'] += int(correct.sum())
                t['choice_rows'] += int(choice.sum()); t['choice_correct'] += int((correct & choice).sum())
                t['nll_sum'] += float(loss.sum())
                t['argmax_disagreement'] += int((winner != fp_winner).sum())
                t['choice_argmax_disagreement'] += int(((winner != fp_winner)&choice).sum())
                for i,c in enumerate(counts.tolist()):
                    ss = scores[i,:c].double().cpu().numpy()
                    ref = fp[i,:c].double().cpu().numpy()
                    dd = distributions[mode]
                    dd['row_mean'].append(float(ss.mean())); dd['row_span'].append(float(np.ptp(ss)))
                    dd['row_absmax'].append(float(np.abs(ss).max()))
                    dd['centered_abs_error_max'].append(float(np.abs((ss-ss.mean())-(ref-ref.mean())).max()))
                    dd['raw_abs_error_max'].append(float(np.abs(ss-ref).max()))
                    dd['unique_fraction'].append(float(len(np.unique(ss))/c))
                    ordered = np.sort(ss)
                    dd['top2_margin'].append(float(ordered[-1]-ordered[-2]) if c>1 else 0.)
                    t['choice_top_tie_rows'] += int(c>1 and np.count_nonzero(ss==ss.max())>1)
                    t['score_min'] = min(t['score_min'],float(ss.min())); t['score_max'] = max(t['score_max'],float(ss.max()))
                    if mode=='bf16' and c>1 and winner[i]!=fp_winner[i] and len(examples)<16:
                        examples.append({'candidates':c,'fp_mean':float(ref.mean()),'fp_span':float(np.ptp(ref)),
                                         'bf_mean':float(ss.mean()),'bf_span':float(np.ptp(ss)),
                                         'fp_winner':int(fp_winner[i]),'bf_winner':int(winner[i]),
                                         'fp_positive':bool(positives[i,fp_winner[i]]),'bf_positive':bool(positives[i,winner[i]]),
                                         'centered_error_max':float(np.abs((ss-ss.mean())-(ref-ref.mean())).max())})
    for mode,t in totals.items():
        t['accuracy'] = t['correct']/t['rows']; t['choice_accuracy'] = t['choice_correct']/t['choice_rows']
        t['nll'] = t['nll_sum']/t['rows']; t['choice_argmax_disagreement_rate'] = t['choice_argmax_disagreement']/t['choice_rows']
        t['choice_top_tie_rate'] = t['choice_top_tie_rows']/t['choice_rows']

    gradients = {}
    # All copies remain eval mode: no dropout exists. This assesses numerical
    # gradient differences, not a training update or train distribution.
    for mode in ('fp32','bf16','bf16_head2_fp32'):
        model.zero_grad(set_to_none=True)
        loss_sum = 0.; n = sum(len(p) for _,p in gradient_samples)
        hook = model.head2.register_forward_hook(fp32_head_hook) if mode=='bf16_head2_fp32' else None
        for batch,p in gradient_samples:
            batch = {k:v.cuda() for k,v in batch.items()}; p=p.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16) if mode!='fp32' else nullcontext():
                scores = model(**batch)
                loss = marginal_loss(scores,p,'none').sum()/n
            loss.backward(); loss_sum += float(loss.detach())
        if hook is not None: hook.remove()
        vector=torch.cat([p.grad.detach().float().flatten().cpu() for p in model.parameters() if p.grad is not None])
        gradients[mode]={'rows':n,'mean_nll':loss_sum,'gradient_norm':float(vector.norm()),
                         'head2_bias_gradient':float(model.head2.bias.grad), 'vector':vector}
    base=gradients['fp32']['vector']
    for mode,item in gradients.items():
        v=item.pop('vector')
        item['cosine_to_fp32']=float(torch.nn.functional.cosine_similarity(base,v,dim=0))
        item['relative_l2_error']=float((v-base).norm()/base.norm())
    if sha(checkpoint)!=original_sha: raise RuntimeError('checkpoint changed during diagnostic')
    return {'checkpoint':str(checkpoint),'checkpoint_sha256':original_sha,
            'selected_epoch':ckpt.get('provenance',{}).get('selected_epoch'),
            'metrics':totals,'distributions':{k:{n:summary(v) for n,v in d.items()} for k,d in distributions.items()},
            'gradient_diagnostic':gradients,'bf16_argmax_change_examples':examples}


def markdown(report):
    lines=['# BC FP32 / BF16 precision diagnostic','',
           'Same frozen checkpoint, same validation rows. No held-out test opened, weights or training source modified. FP32-centered rounding subtracts each row mean **before** output quantization; this is a synthetic isolation test, not production code.','',
           '| checkpoint | mode | NLL | raw accuracy | choice accuracy | choice argmax differs vs FP32 | choice top ties | score range | centered error p95 / max |',
           '|---|---|---:|---:|---:|---:|---:|---|---|']
    for name,r in report['checkpoints'].items():
        for mode,t in r['metrics'].items():
            d=r['distributions'][mode]['centered_abs_error_max']
            lines.append(f"| {name} | {mode} | {t['nll']:.6f} | {t['accuracy']:.3%} | {t['choice_accuracy']:.3%} | {t['choice_argmax_disagreement_rate']:.3%} | {t['choice_top_tie_rate']:.3%} | [{t['score_min']:.4f}, {t['score_max']:.4f}] | {d['p95']:.5f} / {d['max']:.5f} |")
    lines+=['','## Gradient probe on the same validation prefix','',
            '| checkpoint | mode | rows | NLL | gradient norm | head2 bias gradient | cosine to FP32 | relative L2 error |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,r in report['checkpoints'].items():
        for mode,g in r['gradient_diagnostic'].items():
            lines.append(f"| {name} | {mode} | {g['rows']} | {g['mean_nll']:.6f} | {g['gradient_norm']:.5f} | {g['head2_bias_gradient']:.8g} | {g['cosine_to_fp32']:.6f} | {g['relative_l2_error']:.6f} |")
    lines+=['','The candidate marginal softmax loss is invariant to a shared exact real-valued score offset; head2 bias derivative is mathematically zero. BF16 has 7 mantissa bits, so a large common score component can turn small relative differences into tied rounded outputs. Removing offsets **after** BF16 quantization cannot recover lost differences. Both output-rounding and full-network precision effects are reported separately.','',
            'Only best.pt is available for bc-v2-full (epoch 2). This diagnostic cannot directly reproduce the epoch 8 training-loss increase or establish which later tensor caused it. An independent FP32 training run is the causal contrast for deciding whether full-candidate BC helps.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=ROOT/'data/processed/bc-v1')
    p.add_argument('--v1',type=Path,default=ROOT/'models/bc-v1/best.pt')
    p.add_argument('--v2',type=Path,default=ROOT/'models/bc-v2-full/best.pt')
    p.add_argument('--output',type=Path,default=ROOT/'reports/bc-precision-diagnostic.json')
    p.add_argument('--markdown',type=Path,default=ROOT/'reports/bc-precision-diagnostic.md')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--gradient-rows',type=int,default=256)
    args=p.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    files=sorted((args.data/'validation').glob('*.npz'))
    if not files: raise ValueError('validation shards required')
    report={'schema':'oxbot-bc-precision-diagnostic-v1','status':'complete','split':'validation','test_data_opened':False,
            'torch':str(torch.__version__),'device':torch.cuda.get_device_name(0),
            'validation_shards':{str(x.relative_to(args.data)):sha(x) for x in files},
            'script_sha256':sha(__file__),'checkpoints':{}}
    # v2-full was prepared from a new train manifest, but its validation
    # shards must be byte-identical before comparing the two checkpoints.
    v2_data = ROOT / 'data/processed/bc-v2-full'
    if v2_data.is_dir():
        v2_files = sorted((v2_data/'validation').glob('*.npz'))
        common = {x.name: sha(x) for x in files}
        v2_common = {x.name: sha(x) for x in v2_files}
        report['v2_validation_shards_byte_identical'] = common == v2_common
        report['v2_data_manifest_sha256'] = sha(v2_data/'manifest.json')
        if common != v2_common:
            raise RuntimeError('v1 and v2 validation shards differ; refusing mixed comparison')
    else:
        report['v2_validation_shards_byte_identical'] = None
    for name,path in [('bc-v1',args.v1),('bc-v2-full',args.v2)]:
        report['checkpoints'][name]=run(path,files,args)
        print(json.dumps({'checkpoint':name,'metrics':report['checkpoints'][name]['metrics']}),flush=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    args.markdown.write_text(markdown(report),encoding='utf-8')


if __name__=='__main__': main()
