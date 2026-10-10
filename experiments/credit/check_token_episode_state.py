"""R1 trained native token model, fresh real BPE suffix, state/gradient/causality contracts."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'))
from r1_token_keyed_lm import KeyedTokenLM,load
from token_episode_state import consume,revealed_nll,detach_state
RECORD='experiments/results/token_language/curie_r1g2_keyed1_4M_s1_20261008T1840Z.json'
SOURCES=['experiments/credit/check_token_episode_state.py','experiments/credit/token_episode_state.py','experiments/r1_token_keyed_lm.py','experiments/tpp/race_tpp_v5.py']

def sequence(model,ids,cut=None):
    forecast=None;state=None;values=[]
    for j in range(ids.shape[1]):
        if forecast is not None:values.append(revealed_nll(forecast,ids[:,j]))
        forecast,state=consume(model,ids[:,j],state)
        if cut==j:state=detach_state(state)
    return torch.stack(values,1),state

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);start=time.monotonic();record=json.loads((ROOT/RECORD).read_text());args=record['args']
    for p,h in record['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    model=KeyedTokenLM(args['d'],args['modes'],args['layers'],args['heads'],args['dk'],args['keyed'],args['dropout']);checkpoint=ROOT/record['checkpoint'];model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
    data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');ids=torch.tensor(np.array(data[args['train_tokens']:args['train_tokens']+17],dtype=np.int64))[None];x=ids[:,:-1];y=ids[:,1:];h,e=model.states(x);logits=model.logits(h,e,x,list(range(16)));reference=torch.nn.functional.cross_entropy(logits.reshape(-1,50257),y.reshape(-1),reduction='none').reshape(1,16)
    actual,state=sequence(model,ids);score_error=float((actual-reference).abs().max());assert score_error<2e-5,score_error
    params=tuple(model.parameters());g=torch.autograd.grad(reference.mean(),params,allow_unused=True);other=torch.autograd.grad(actual.mean(),params,allow_unused=True);gradient_error=0.
    for p,u,v in zip(params,g,other):
        u=torch.zeros_like(p) if u is None else u;v=torch.zeros_like(p) if v is None else v;gradient_error=max(gradient_error,float((u-v).abs().max()))
    assert gradient_error<2e-5,gradient_error
    with torch.no_grad():
        carried,_=sequence(model,ids,cut=8);detach_error=float((carried-actual).abs().max());assert detach_error<2e-5
        changed=ids.clone();changed[:,9:]=(changed[:,9:]+1)%50257;suffix,_=sequence(model,changed);causal_error=float((suffix[:,:8]-actual[:,:8]).abs().max());assert causal_error==0
        pending,_=consume(model,ids[:,0]);before=revealed_nll(pending,ids[:,1]).clone();model.bias.add_(.01);after=revealed_nll(pending,ids[:,1]);assert torch.equal(before,after)
    result=dict(status='completed',tag=a.tag,battle='R1',training=False,real_gpt2_targets=16,fresh_train_suffix_start=args['train_tokens'],score_error=score_error,all_parameter_gradient_error=gradient_error,detach_preserves_state_error=detach_error,future_suffix_causality_error=causal_error,issued_forecast_preserved_after_update=True,addressed_tokens=len(state['keys']),persistent_key_values=sum(v.numel() for v,_ in state['keys'].values()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),wall_s=time.monotonic()-start,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},scope='Available trained keyed-token member, GPT-2 vocabulary and fresh TRAIN suffix. Lazy per-token key transport is an exact frozen-parameter representation of the recorded read; all present keys still scored and full vocabulary output paid. This is a member-specific contract, not proof of internal hard races or all sparse-family gradients. No public scoring slice read or new fitting.')
    output=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
