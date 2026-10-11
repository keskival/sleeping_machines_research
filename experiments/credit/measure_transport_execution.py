"""Matched acquisition boundary: exact route credit vs transported+factual credit."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM
from measure_token_forward_credit import verify
from token_issued_route_pool import issue_with_history
from token_forward_credit_vectorized import packet
from token_exact_world_readouts import replay
from token_optimizer_transport import replay_transport
from token_flop_accounting import TokenFlopCounter


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','measure'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    d=verify('curie_long_token_credit_20261011_v1_measure');verify('curie_token_optimizer_transport_20261011_v1_measure')
    if a.mode=='measure':verify('curie_transport_execution_20261011_v1_smoke')
    labels=ROOT/d['checkpoint'];assert hashlib.sha256(labels.read_bytes()).hexdigest()==d['checkpoint_sha256'];saved=torch.load(labels,weights_only=True)['prefixes']
    configuration=json.loads((ROOT/SOURCE).read_text());c=configuration['args'];checkpoint=ROOT/configuration['checkpoint']
    m=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);m.load_state_dict(torch.load(checkpoint,weights_only=True));m.requires_grad_(False);m.eval()
    count,repeats,targets=(1,1,16) if a.mode=='smoke' else (4,3,48);rows=[];common=[]
    for i,s in enumerate(saved[:count]):
        begin=time.monotonic()
        with TokenFlopCounter(display=False) as counter:
            issued=issue_with_history(m,s['tokens'][:64],s['tokens'][:64]);p=packet(m,s['tokens'][:64+targets],index=63)
        assert issued['receivers']==s['receivers'] and torch.equal(issued['actions'],s['actions'])
        common.append(dict(prefix=i,wall_s=time.monotonic()-begin,supported_flops=counter.get_total_flops()))
        for repeat in range(repeats):
            action=int(torch.randint(len(s['receivers']),(),generator=torch.Generator().manual_seed(101000+i*10+repeat)))
            for arm in (['exact','transport_corrected'] if repeat%2==0 else ['transport_corrected','exact']):
                begin=time.monotonic()
                with TokenFlopCounter(display=False) as counter:
                    if arm=='exact':
                        result=replay(p,s['receivers'],s['actions'],.01);scores=result['scores'];worlds=len(scores)
                        error=float((scores-s['true_scores'][:,:targets]).abs().max());assert error<1e-10
                    else:
                        result=replay_transport(p,s['receivers'],s['actions'],.01,'diagonal');scores=result['scores'];worlds=1
                        if action!=0:
                            factual=replay(p,[s['receivers'][action]],s['actions'][action:action+1],.01)
                            observed=factual['scores'][0];worlds+=1
                        else:observed=result['reference_scores']
                        error=float((observed-s['true_scores'][action,:targets]).abs().max());assert error<1e-10
                        reference_error=float((result['reference_scores']-s['true_scores'][0,:targets]).abs().max());assert reference_error<1e-10
                        pi=torch.full((len(scores),),1/len(scores));h=scores.mean(1)-scores[0].mean()
                        contrast=observed.mean()-result['reference_scores'].mean()
                        correction=pi*(h-(pi*h).sum())+(contrast-h[action])*(torch.nn.functional.one_hot(torch.tensor(action),len(scores))-pi)
                        assert torch.isfinite(correction).all()
                rows.append(dict(prefix=i,repeat=repeat,arm=arm,action=action,wall_s=time.monotonic()-begin,supported_flops=counter.get_total_flops(),actual_worlds=worlds,actual_score_error=error))
        print(json.dumps(dict(prefix=i,completed_repeats=repeats)),flush=True)
    summaries=[]
    for arm in ['exact','transport_corrected']:
        stage=sum(float(np.median([r['wall_s'] for r in rows if r['arm']==arm and r['prefix']==i])) for i in range(count))
        work=sum(float(np.median([r['supported_flops'] for r in rows if r['arm']==arm and r['prefix']==i])) for i in range(count))
        summaries.append(dict(arm=arm,median_stage_wall_s=stage,median_stage_supported_flops=work,whole_pipeline_wall_s=stage+sum(r['wall_s'] for r in common),whole_pipeline_supported_flops=work+sum(r['supported_flops'] for r in common)))
    paths=['experiments/credit/measure_transport_execution.py','experiments/credit/token_optimizer_transport.py','experiments/credit/token_exact_world_readouts.py','experiments/credit/token_issued_route_pool.py','experiments/credit/token_forward_credit_vectorized.py','experiments/credit/token_flop_accounting.py']
    result=dict(status='completed',battle='R1',tag=a.tag,mode=a.mode,gate=True,rows=rows,common=common,summaries=summaries,whole_pipeline_speedup=summaries[0]['whole_pipeline_wall_s']/summaries[1]['whole_pipeline_wall_s'],supported_work_ratio=summaries[1]['whole_pipeline_supported_flops']/summaries[0]['whole_pipeline_supported_flops'],whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_labels_sha256=d['checkpoint_sha256'],source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Matched fixed carrier/prefix candidate acquisition and proper-token route universe, three rotated repeats per prefix in measurement. Exact computes all independent learner costs; approximation transports all Adam worlds plus executes selected factual learner and real reference, full residual issued-uniform logit update paid. Actual/reference losses exact, predictor approximate (accuracy recorded separately). Whole current acquisition pipeline includes shared source context/normalization/keys/candidate discovery. No task fitting/whole-model/depth/energy or matched-quality learning-speed claim. All repeated execution work and oracle/source pretraining additional; supported FLOPs omit unsupported work included in wall. No DEV/TEST.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(summaries=summaries,whole_pipeline_speedup=result['whole_pipeline_speedup'])),flush=True)
if __name__=='__main__':main()
