"""R1 real GPT-2-token stream carry and causal forecast contract; no fitting."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
import horizon_token_language_engine as engine
from sleeping_machines.chunk_gather_counterfactual_episodes import token_features,detach
RECORD='experiments/results/token_language/curie_data_growth_tokens_1m_b64_c16_p24_s6_20261005_v1.json'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);start=time.monotonic();record=json.loads((ROOT/RECORD).read_text());args=SimpleNamespace(**record['args']);sources=dict(record['identity']['source_sha256'])
    for p,expected in sources.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==expected,p
    tokens=engine.load_tokens(ROOT/args.train_file);counts=engine.LaneTokens(tokens,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True)
    torch.manual_seed(args.seed);model=engine.Model(args,order,counts);checkpoint=Path(record['best_checkpoint']);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
    # Fresh TRAIN suffix starts after the complete warm-start fitting interval.
    ids=torch.tensor(np.array(tokens[args.train_tokens:args.train_tokens+33],dtype=np.int64))[None];x=ids[:,:32];y=ids[:,1:]
    def features(data,state=None):
        return token_features(model.core,data,state,torch.Generator().manual_seed(181),route_credit='none',deterministic=True,eos=50256,free_bias=args.free_bias,temperature=args.temperature,compiled=False)
    full,fs,_=features(x);left,ls,_=features(x[:,:16]);right,rs,_=features(x[:,16:],ls);joined=torch.cat((left,right),1)
    fe=float((full-joined).abs().max());state_error=max(float((a-b).abs().max()) for a,b in zip(fs['mem'],rs['mem']));assert fe<1e-5 and state_error<1e-5
    parameters=tuple(model.parameters());loss=model.readout.nll(full,y).mean();other=model.readout.nll(joined,y).mean();g=torch.autograd.grad(loss,parameters,allow_unused=True);h=torch.autograd.grad(other,parameters,allow_unused=True);ge=0.
    for p,u,v in zip(parameters,g,h):
        u=torch.zeros_like(p) if u is None else u;v=torch.zeros_like(p) if v is None else v;ge=max(ge,float((u-v).abs().max()))
    assert ge<1e-5,ge
    # Detachment changes credit horizon, preserving carried numerical state.
    with torch.no_grad():
        left,st,_=features(x[:,:16]);right,ds,_=features(x[:,16:],detach(st));de=float((torch.cat((left,right),1)-full).abs().max());assert de<1e-5
        changed=x.clone();changed[:,16:]=(changed[:,16:]+1)%50257;cx,_,_=features(changed);causal=float((cx[:,:16]-full[:,:16]).abs().max());assert causal==0
        # Issued distribution uses a cached feature; later readout changes must
        # not re-score that forecast. Store the score/distribution before update.
        forecast_nll=model.readout.nll(left[:,-1:],y[:,15:16]).detach().clone()
    result=dict(status='completed',tag=a.tag,battle='R1',training=False,real_gpt2_targets=32,train_suffix_start=args.train_tokens,forward_error=fe,numerical_state_error=state_error,all_parameter_gradient_error=ge,detach_preserves_numerical_state_error=de,future_suffix_causality_error=causal,cached_forecast_nll=float(forecast_nll),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),dataset_sha256=record['identity']['train_sha256'],wall_s=time.monotonic()-start,source_sha256={**sources,'experiments/credit/check_language_episode_state.py':hashlib.sha256((ROOT/'experiments/credit/check_language_episode_state.py').read_bytes()).hexdigest(),'sleeping_machines/chunk_gather_counterfactual_episodes.py':hashlib.sha256((ROOT/'sleeping_machines/chunk_gather_counterfactual_episodes.py').read_bytes()).hexdigest()},scope='Proper GPT-2 tokens, saved native sparse/temporal model, contiguous TRAIN suffix, no DEV/public validation/TEST. Full factual gradients with retained graph; detachment truncates credit without resetting state. No within-episode optimizer update or learned-credit efficiency claim.')
    output=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
