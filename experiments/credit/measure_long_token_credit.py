"""R1 longer-horizon credit calibration before delayed-policy development."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,load
from measure_token_forward_credit import verify
from token_issued_route_pool import issue_with_history
from token_forward_credit_vectorized import packet
from token_persistent_credit import joint_replay_carry
from token_exact_world_readouts import replay
from token_flop_accounting import TokenFlopCounter


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','measure'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    assert verify('curie_token_world_readouts_20261011_v1_contract')['gate']
    if a.mode=='measure':assert verify('curie_long_token_credit_20261011_v1_smoke')['status']=='completed'
    count,targets=(1,16) if a.mode=='smoke' else (4,48)
    source=json.loads((ROOT/SOURCE).read_text());c=source['args'];checkpoint=ROOT/source['checkpoint']
    m=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);m.load_state_dict(torch.load(checkpoint,weights_only=True));m.requires_grad_(False);m.eval()
    data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');offset=c['train_tokens']+131072
    summaries=[];saved=[];work_rows=[];worlds=updates=0
    for i in range(count):
        tokens=torch.tensor(np.array(data[offset+i*256:offset+i*256+64+targets],dtype=np.int64))
        begin=time.monotonic()
        with TokenFlopCounter(display=False) as counter:
            issued=issue_with_history(m,tokens[:64],tokens[:64]);p=packet(m,tokens,index=63)
        carrier_cost=counter.get_total_flops();carrier_time=time.monotonic()-begin
        begin=time.monotonic()
        with TokenFlopCounter(display=False) as counter:true=replay(p,issued['receivers'],issued['actions'],.01)
        teacher_cost=counter.get_total_flops();teacher_time=time.monotonic()-begin
        begin=time.monotonic()
        with TokenFlopCounter(display=False) as counter:projection=joint_replay_carry(p,issued['receivers'],issued['actions'],0,0,.01)
        reference_cost=counter.get_total_flops();reference_time=time.monotonic()-begin
        assert float((true['scores'][0]-projection['actual_scores'][0]).abs().max())<1e-10
        d=true['scores'].mean(1);d=d-d[0]
        static=projection['frozen_scores'].mean(1);static=static-static[0]
        reference=projection['projected_scores'].mean(1);reference=reference-reference[0]
        early=true['scores'][:,:8].sum(1)/targets;early=early-early[0]
        pi=torch.full_like(d,1/len(d));score_norm=1-2*pi+pi.square().sum()
        for name,h in [('zero',torch.zeros_like(d)),('first8_exact_no_tail',early),('static_full',static),('reference_full',reference)]:
            residual=d-h;g=pi*(residual-(pi*residual).sum())
            variance=float((pi*residual.square()*score_norm).sum()-g.square().sum())
            summaries.append(dict(prefix=i,arm=name,mse=float(residual.square().mean()),conditional_logit_variance=variance))
        first=int(true['scores'][:,:8].mean(1).argmin());full=int(d.argmin())
        work_rows.append(dict(prefix=i,routes=len(d),carrier_supported_flops=carrier_cost,carrier_wall_s=carrier_time,teacher_supported_flops=teacher_cost,teacher_wall_s=teacher_time,reference_supported_flops=reference_cost,reference_wall_s=reference_time,first8_oracle_route=first,full_oracle_route=full,first8_choice_full_regret=float(d[first]-d[full]),true_future_tail_rms=float((d-early).square().mean().sqrt())))
        worlds+=true['actual_worlds']+projection['actual_worlds'];updates+=true['readout_updates']+projection['readout_updates']
        saved.append(dict(tokens=tokens,receivers=issued['receivers'],actions=issued['actions'],true_scores=true['scores'],projected_scores=projection['projected_scores'],static_scores=projection['frozen_scores']))
        print(json.dumps(work_rows[-1]),flush=True)
    rows=[dict(arm=name,mse=float(np.mean([r['mse'] for r in summaries if r['arm']==name])),conditional_logit_variance=float(np.mean([r['conditional_logit_variance'] for r in summaries if r['arm']==name]))) for name in ['zero','first8_exact_no_tail','static_full','reference_full']]
    state=out.with_suffix('.pt');torch.save(dict(prefixes=saved),state)
    paths=['experiments/credit/measure_long_token_credit.py','experiments/credit/token_issued_route_pool.py','experiments/credit/token_persistent_credit.py','experiments/credit/token_exact_world_readouts.py','experiments/credit/token_forward_credit_vectorized.py','experiments/credit/token_flop_accounting.py']
    result=dict(status='completed',battle='R1',tag=a.tag,mode=a.mode,rows=rows,per_prefix=summaries,work=work_rows,context=64,targets_per_prefix=targets,train_offset=offset,actual_worlds=worlds,readout_updates=updates,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checkpoint=str(state.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Four fresh properly tokenized TRAIN prefixes/48futuretargets/64context in measurement. Exact independent online lexical Adam worlds for all prefix-issued hard writes; shared no-op reference projection. Early8 exact contrast divided by full target count supplies zero-tail diagnostic, same cost units as full-horizon contrasts. Uniform issued-logit variance, one observed future per prefix; no future-distribution/controller-parameter variance or utility/benchmark claim. Full carrier/projection/scoring/readout work charged, source pretraining/contracts/smokes additional; unsupported arithmetic included in wall but omitted from supported FLOPs. Saved TRAIN labels may support later registered boundary-critic fitting, no public DEV/TEST.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(rows=rows,whole_wall_s=result['whole_wall_s'])),flush=True)
if __name__=='__main__':main()
