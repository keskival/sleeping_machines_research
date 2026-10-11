"""Replay paid exact TRAIN worlds to calibrate economical optimizer transport."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM
from measure_token_forward_credit import verify
from token_forward_credit_vectorized import packet
from token_optimizer_transport import replay_transport
from token_flop_accounting import TokenFlopCounter
SOURCE_RESULT='curie_long_token_credit_20261011_v1_measure'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','measure'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    assert verify('curie_token_optimizer_transport_20261011_v1_contract')['gate']
    if a.mode=='measure':assert verify('curie_token_optimizer_transport_20261011_v1_smoke')['status']=='completed'
    source=verify(SOURCE_RESULT);labels=ROOT/source['checkpoint'];assert hashlib.sha256(labels.read_bytes()).hexdigest()==source['checkpoint_sha256']
    saved=torch.load(labels,weights_only=True)['prefixes'];count,targets=(1,16) if a.mode=='smoke' else (4,48)
    configuration=json.loads((ROOT/SOURCE).read_text());c=configuration['args'];checkpoint=ROOT/configuration['checkpoint'];assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==source['source_checkpoint_sha256']
    m=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);m.load_state_dict(torch.load(checkpoint,weights_only=True));m.requires_grad_(False);m.eval()
    rows=[];work=[];predictions=[]
    for i,s in enumerate(saved[:count]):
        begin=time.monotonic()
        with TokenFlopCounter(display=False) as counter:p=packet(m,s['tokens'][:64+targets],index=63)
        work.append(dict(prefix=i,arm='carrier_replay',wall_s=time.monotonic()-begin,supported_flops=counter.get_total_flops()))
        true=s['true_scores'][:,:targets].mean(1);d=true-true[0]
        for name,score in [('static',s['static_scores'][:,:targets]),('reference',s['projected_scores'][:,:targets])]:
            h=score.mean(1);h=h-h[0];rows.append(dict(prefix=i,arm=name,mse=float((d-h).square().mean())))
        for curvature in ['zero','diagonal']:
            begin=time.monotonic()
            with TokenFlopCounter(display=False) as counter:t=replay_transport(p,s['receivers'],s['actions'],.01,curvature)
            elapsed=time.monotonic()-begin;parity=float((t['reference_scores']-s['true_scores'][0,:targets]).abs().max());assert parity<1e-10
            h=t['scores'].mean(1);h=h-h[0];pi=torch.full_like(d,1/len(d));residual=d-h;g=pi*(residual-(pi*residual).sum())
            score_norm=1-2*pi+pi.square().sum();variance=float((pi*residual.square()*score_norm).sum()-g.square().sum())
            rows.append(dict(prefix=i,arm='transport_'+curvature,mse=float(residual.square().mean()),conditional_logit_variance=variance))
            work.append(dict(prefix=i,arm='transport_'+curvature,wall_s=elapsed,supported_flops=counter.get_total_flops(),reference_parity_error=parity,actual_worlds=1,transported_worlds=len(d),reference_updates=targets,transported_updates=len(d)*targets))
            predictions.append(dict(prefix=i,curvature=curvature,scores=t['scores']))
        print(json.dumps([r for r in rows if r['prefix']==i]),flush=True)
    summary=[dict(arm=name,mse=float(np.mean([r['mse'] for r in rows if r['arm']==name]))) for name in ['static','reference','transport_zero','transport_diagonal']]
    state=out.with_suffix('.pt');torch.save(dict(predictions=predictions),state)
    paths=['experiments/credit/measure_token_optimizer_transport.py','experiments/credit/token_optimizer_transport.py','experiments/credit/token_issued_credit_geometry.py','experiments/credit/token_forward_credit_vectorized.py','experiments/credit/token_flop_accounting.py']
    result=dict(status='completed',battle='R1',tag=a.tag,mode=a.mode,rows=summary,per_prefix=rows,work=work,source_result=SOURCE_RESULT,source_labels_sha256=source['checkpoint_sha256'],targets_per_prefix=targets,prefixes=count,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint=str(state.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Reuses paid all-route exact48-target TRAIN learner labels; no new independent oracle fit. Common native carrier replay plus one real reference lexical Adam and all alternative optimizer-state transport/prediction paid separately. Zero/diagonal curvature are declared approximations, reference score parity enforced at real50257 vocabulary. Arithmetic is supported FLOPs, unsupported work included in wall. Oracle acquisition/source pretraining/contracts/smoke additional; no controller fitting/DEV/TEST or task/depth/whole-model learning-efficiency claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(rows=summary,whole_wall_s=result['whole_wall_s'])),flush=True)
if __name__=='__main__':main()
