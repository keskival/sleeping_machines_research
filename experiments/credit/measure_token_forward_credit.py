"""R1 proper-token TRAIN calibration: exact worlds, static and reference learning."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,load
from token_forward_credit import packet,joint_replay
from token_exact_world_readouts import replay
TAG='curie_token_forward_credit_calibration_20261011_v2'
SOURCES=['experiments/credit/measure_token_forward_credit.py','experiments/credit/token_exact_world_readouts.py','experiments/credit/token_forward_credit.py','experiments/credit/token_episode_state.py','experiments/r1_token_keyed_lm.py','experiments/tpp/race_tpp_v5.py']

def verify(name):
    d=json.loads((ROOT/f'experiments/results/credit/{name}.json').read_text());assert d['status']=='completed'
    for p,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    return d

def candidates(tokens,write,heads,dk):
    addresses=[]
    for token in reversed(tokens.tolist()):
        if token==50256:break
        if token not in addresses:addresses.append(token)
        if len(addresses)==8:break
    receivers=[int(tokens[-1])];actions=[write.new_zeros(heads,dk)]
    direction=write.reshape(heads,dk);direction=direction/direction.norm(dim=1,keepdim=True).clamp_min(1e-12)
    for r in addresses:
        for h in range(heads):
            for sign in [1.,-1.]:
                message=write.new_zeros(heads,dk);message[h]=sign*.25*direction[h]
                receivers.append(r);actions.append(message)
    return receivers,torch.stack(actions)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','measure'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    for name in ['curie_token_forward_credit_20261011_v1_contract','curie_token_world_readouts_20261011_v1_contract']:assert verify(name)['gate']
    if a.mode=='measure':verify(TAG+'_smoke')
    source=json.loads((ROOT/SOURCE).read_text());c=source['args'];checkpoint=ROOT/source['checkpoint'];cp_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);model.load_state_dict(torch.load(checkpoint,weights_only=True));model.requires_grad_(False);model.eval()
    data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');offset=c['train_tokens']+8192
    count,context,targets=(1,16,4) if a.mode=='smoke' else (8,256,8)
    rows=[];saved=[];work=[];total_worlds=readout_updates=0
    for i in range(count):
        tokens=torch.tensor(np.array(data[offset+i*targets:offset+i*targets+context+targets],dtype=np.int64));write=[]
        hook=model.key_w.register_forward_hook(lambda module,args,result:write.append(result.detach()))
        tick=time.monotonic()
        try:
            with FlopCounterMode(display=False) as counter:p=packet(model,tokens,index=context-1)
        finally:hook.remove()
        carrier_cost=counter.get_total_flops();carrier_wall=time.monotonic()-tick
        receivers,actions=candidates(tokens[:context],write[context-1],model.heads,model.dk)
        tick=time.monotonic()
        with FlopCounterMode(display=False) as counter:true=replay(p,receivers,actions,.01)
        teacher_cost=counter.get_total_flops();teacher_wall=time.monotonic()-tick
        tick=time.monotonic()
        with FlopCounterMode(display=False) as counter:projected=joint_replay(p,receivers,actions,0,0,.01)
        projection_cost=counter.get_total_flops();projection_wall=time.monotonic()-tick
        error=float((true['scores'][0]-projected['actual_scores'][0]).abs().max());assert error<1e-10
        total_worlds+=true['actual_worlds']+projected['actual_worlds'];readout_updates+=true['readout_updates']+projected['readout_updates']
        d=true['scores'].mean(1);d=d-d[0];h_static=projected['frozen_scores'].mean(1);h_static=h_static-h_static[0];h_reference=projected['projected_scores'].mean(1);h_reference=h_reference-h_reference[0]
        pi=torch.full_like(d,1./len(d));score_norm=1.-2*pi+pi.square().sum()
        for name,h in [('zero',torch.zeros_like(d)),('forward_static',h_static),('reference_learner',h_reference)]:
            residual=d-h;g=pi*(residual-(pi*residual).sum());variance=float((pi*residual.square()*score_norm).sum()-g.square().sum())
            rows.append(dict(window=i,arm=name,mse=float(residual.square().mean()),contrast_energy=float(d.square().mean()),conditional_policy_logit_variance=variance))
        # Pure adaptation headroom, separate from credit and policy fitting.
        frozen_baseline=float(projected['frozen_scores'][0].mean());adaptive_baseline=float(true['scores'][0].mean())
        work.append(dict(window=i,candidate_routes=len(receivers),context=context,targets=targets,carrier_supported_flops=carrier_cost,carrier_wall_s=carrier_wall,teacher_supported_flops=teacher_cost,teacher_wall_s=teacher_wall,reference_supported_flops=projection_cost,reference_wall_s=projection_wall,frozen_baseline_nll=frozen_baseline,adaptive_baseline_nll=adaptive_baseline,reference_parity_error=error,eos_in_targets=int((p.targets==50256).sum())))
        saved.append(dict(tokens=tokens,receivers=receivers,actions=actions,true_scores=true['scores'],final_world_z=true['z'],static_scores=projected['frozen_scores'],reference_scores=projected['projected_scores']))
        print(json.dumps(work[-1]),flush=True)
    summaries=[]
    for name in ['zero','forward_static','reference_learner']:
        subset=[r for r in rows if r['arm']==name]
        mse=float(np.mean([r['mse'] for r in subset]))
        energy=float(np.mean([r['contrast_energy'] for r in subset]))
        variance=float(np.mean([r['conditional_policy_logit_variance'] for r in subset]))
        summaries.append(dict(arm=name,mse=mse,error_over_zero=mse/energy if energy else None,
                              mean_conditional_policy_logit_variance=variance))

    checkpoint=out.with_suffix('.pt');torch.save(dict(windows=saved),checkpoint)
    result=dict(status='completed',battle='R1',tag=a.tag,mode=a.mode,rows=summaries,per_window=rows,work=work,source_model_result=SOURCE,source_checkpoint_sha256=cp_sha,train_token_offset=offset,context_tokens=context,scored_targets=count*targets,teacher_worlds=total_worlds,readout_updates=readout_updates,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},scope='Proper GPT-2 vocabulary50257/native pretrained2-layer source, fresh contiguous TRAIN targets with256 preceding context in measurement; no public/TEST scoring. Up to8 observed current-document token addresses, one global no-op and8 nonzero messages per address; messages use normalized produced prefix key-write direction in each of4heads, independent of future targets. Exact all-world learner costs, static/reference projections, independent64-dimensional lexical readout/Adam state per world. All carrier/context replay, full normalized outputs/scans, independent learners and additional duplicate reference learning charged; source4M pretraining and numerical gates additional. Fast/native state reset per measurement window and observed EOS: this is calibration/headroom, not persistent online or actor-policy training. Uniform issued policy-logit covariance, not native/carrier parameter variance; one observed continuation per context, no independent-future variance claim. Supported FLOPs omit unsupported scalar/special/optimizer arithmetic, included in wall. Saved paid TRAIN labels may support later declared fitting; no benchmark/depth/hardware claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
