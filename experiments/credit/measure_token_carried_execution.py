"""R1 whole carried factual/reference token execution, exact legacy vs vectorized."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,load
from token_forward_credit import packet as legacy
from token_forward_credit_vectorized import packet as vectorized
from token_persistent_credit import joint_replay_carry
from measure_token_forward_credit import candidates
from token_flop_accounting import TokenFlopCounter
SOURCES=['experiments/credit/measure_token_carried_execution.py','experiments/credit/token_forward_credit_vectorized.py','experiments/credit/token_episode_state_vectorized.py','experiments/credit/token_persistent_credit.py','experiments/credit/token_forward_credit.py','experiments/credit/token_episode_state.py','experiments/credit/token_flop_accounting.py','experiments/credit/measure_token_forward_credit.py','experiments/r1_token_keyed_lm.py','experiments/tpp/race_tpp_v5.py']
TAG='curie_token_carried_execution_20261011_v1'

def verify(name):
    d=json.loads((ROOT/f'experiments/results/credit/{name}.json').read_text());assert d['status']=='completed' and d.get('gate',True)
    for p,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    return d


def run(model,tokens,context,targets,fn):
    start=time.monotonic();carrier=learner=None;scores=[];reference=[];choices=[];updates=worlds=projections=0;windows=[]
    with TokenFlopCounter() as work:
        for i in range(targets//8):
            begin=0 if i==0 else context-1+i*8
            index=context-1 if i==0 else 0
            end=context+(i+1)*8;current=tokens[begin:end];writes=[]
            hook=model.key_w.register_forward_hook(lambda module,args,result:writes.append(result.detach()))
            try:p=fn(model,current,index=index,initial_state=carrier)
            finally:hook.remove()
            observed=tokens[:context+i*8];rs,actions=candidates(observed,writes[index],model.heads,model.dk)
            # Positive-support fixed probe law, chosen from past/prefix only.
            selected=int(np.random.default_rng(96000000+i).integers(len(rs)))
            out=joint_replay_carry(p,rs,actions,selected,0,.01,initial_learner=learner)
            scores.extend(out['actual_scores'][0].tolist());reference.extend(out['actual_scores'][1].tolist())
            choices.append(selected);worlds+=out['actual_worlds'];updates+=out['readout_updates'];projections+=len(rs)*8
            carrier=out['factual_carrier'];learner=out['factual_learner']
            windows.append(dict(window=i,candidates=len(rs),action=selected,live_key_slots=0 if carrier is None else len(carrier['keys']),actual_worlds=out['actual_worlds']))
    return dict(nll=float(np.mean(scores)),targets=len(scores),wall_s=time.monotonic()-start,supported_flops=work.get_total_flops(),actual_worlds=worlds,readout_updates=updates,projected_route_targets=projections,scores=scores,reference_scores=reference,choices=choices,windows=windows),learner,carrier


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','measure'],required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    for gate in ['curie_token_persistent_credit_20261011_v1_contract','curie_token_vectorized_transport_20261011_v1_contract','curie_token_flop_accounting_20261011_v1_contract']:verify(gate)
    if a.mode=='measure':verify(TAG+'_smoke')
    source=json.loads((ROOT/SOURCE).read_text());c=source['args'];checkpoint=ROOT/source['checkpoint'];model=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);model.load_state_dict(torch.load(checkpoint,weights_only=True));model.requires_grad_(False);model.eval()
    context,targets=(16,16) if a.mode=='smoke' else (256,128)
    data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');offset=c['train_tokens']+16384;tokens=torch.tensor(np.array(data[offset:offset+context+targets],dtype=np.int64));rows=[];states=[]
    for name,fn in [('legacy','legacy'),('vectorized','vectorized')]:
        row,learner,carrier=run(model,tokens,context,targets,legacy if fn=='legacy' else vectorized);row['arm']=name;rows.append(row);states.append((learner,carrier));print(json.dumps({k:v for k,v in row.items() if k not in ['scores','reference_scores','windows']}),flush=True)
    left,right=rows;errors=dict(factual_scores=float(np.max(np.abs(np.array(left['scores'])-right['scores']))),reference_scores=float(np.max(np.abs(np.array(left['reference_scores'])-right['reference_scores']))),final_z=float((states[0][0]['z']-states[1][0]['z']).abs().max()))
    assert left['choices']==right['choices'] and left['actual_worlds']==right['actual_worlds'] and max(errors.values())<1e-10,errors
    for k,v in states[0][0]['optimizer']['state'].items():
        for n in ['step','exp_avg','exp_avg_sq']:assert float((v[n]-states[1][0]['optimizer']['state'][k][n]).abs().max())<1e-10
    result=dict(status='completed',battle='R1',tag=a.tag,mode=a.mode,gate=True,rows=rows,errors=errors,whole_execution_speedup=left['wall_s']/right['wall_s'],source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),train_token_offset=offset,context=context,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},scope='One contiguous proper GPT-2 TRAIN stream, same pretrained2layer model, one initial256-token context, eight-target credit windows with numerical/key/actual fast-vector/Adam carry and EOS reset. Identical positive-support uniform hard-write probes; no actor fitting. Complete current pipeline boundary includes source carrier/full normalized output, all active-key scores, reference/counterfactual projections, actual learner Adam updates, state/copy/metadata and incoming snapshot cloning. No context replay after initial context. Both arms include unused zero-fast-state frozen projections equally; those are not current-state static controls. Custom2VD mv counts included; unsupported scalar/special/optimizer arithmetic omitted from supported FLOPs and included in wall. One execution observation, not learned-credit quality/whole-model/depth/hardware efficiency claim; source pretraining/numerical gates/calibration additional.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
