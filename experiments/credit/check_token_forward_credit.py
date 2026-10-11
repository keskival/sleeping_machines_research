"""R1 independent native proper-token replay contracts for forward-derived credit."""
import argparse,hashlib,json,math,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'))
from r1_token_keyed_lm import KeyedTokenLM,load
from token_episode_state import consume
from token_forward_credit import packet,joint_replay,projected_costs
SOURCES=['experiments/credit/check_token_forward_credit.py','experiments/credit/token_forward_credit.py','experiments/credit/token_episode_state.py','experiments/r1_token_keyed_lm.py','experiments/tpp/race_tpp_v5.py']
SOURCE='experiments/results/token_language/curie_r1g2_keyed1_4M_s1_20261008T1840Z.json'

def independent(model,tokens,receiver,action,lr,initial_z=None,eos_token=50256):
    state=None;captured=[];hook=model.query_w.register_forward_hook(lambda module,args,result:captured.append(result.detach()))
    initial=model.embed.weight.new_zeros(model.embed.embedding_dim) if initial_z is None else initial_z.detach()
    def learner(value):
        z=torch.nn.Parameter(value.clone());return z,torch.optim.Adam([z],lr=lr)
    z,opt=learner(initial);scores=[];gradients=[]
    try:
        for j in range(len(tokens)-1):
            if int(tokens[j])==eos_token:state=None;z,opt=learner(torch.zeros_like(initial))
            with torch.no_grad():
                forecast,state=consume(model,tokens[j:j+1],state,parameter_version=0)
                if j==3 and bool((action!=0).any()):
                    current=state['position']-1;keys=dict(state['keys'])
                    value=model.embed.weight.new_zeros(action.shape) if receiver not in keys else keys[receiver][0]*torch.exp(-model.log_rate.exp()*(current-keys[receiver][1]))[:,None]
                    keys[receiver]=(value+action,current);state=dict(state,keys=keys)
                    query=captured[-1].reshape(model.heads,model.dk)
                    beta=(query*action).sum()/math.sqrt(model.dk)
                    bonus=F.one_hot(torch.tensor(receiver),len(model.embed.weight)).to(model.embed.weight.dtype)*beta
                    forecast=dict(forecast,log_probabilities=(forecast['log_probabilities']+bonus).log_softmax(-1))
            if j>=3:
                opt.zero_grad();logits=forecast['log_probabilities'][0]+model.embed.weight.detach()@z
                loss=F.cross_entropy(logits[None],tokens[j+1:j+2]);scores.append(loss.detach())
                loss.backward();gradients.append(z.grad.detach().clone());opt.step()
    finally:hook.remove()
    if int(tokens[-1])==eos_token:z,opt=learner(torch.zeros_like(initial))
    return dict(scores=torch.stack(scores),gradients=gradients,z=z.detach().clone(),optimizer=opt.state_dict())


def check(model,tokens,eos_token):
    actions=torch.zeros(3,model.heads,model.dk);actions[0,0,0]=.25;actions[1,-1,-1]=-.25
    receivers=[int(tokens[2]),len(model.embed.weight)-1,int(tokens[3])];p=packet(model,tokens,eos_token=eos_token)
    actual=joint_replay(p,receivers,actions,0,2,.01,diagnostics=True)
    references=[independent(model,tokens,receivers[k],actions[k],.01,eos_token=eos_token) for k in [2,0]]
    errors={}
    for k,world in enumerate(references):
        row=1 if k==0 else 0
        errors[f'scores{k}']=float((actual['actual_scores'][row]-world['scores']).abs().max())
        errors[f'z{k}']=float((actual['final_z'][k]-world['z']).abs().max())
        errors[f'gradients{k}']=max(float((v-actual['gradients'][j][k]).abs().max()) for j,v in enumerate(world['gradients']))
        keys=world['optimizer']['state'].keys();assert actual['optimizers'][k]['state'].keys()==keys
        errors[f'adam{k}']=max([float((world['optimizer']['state'][n][field]-actual['optimizers'][k]['state'][n][field]).abs().max()) for n in keys for field in ['step','exp_avg','exp_avg_sq']]+[0.])
    # Reference row of projected loss is the independently learning actual reference.
    errors['reference_projection']=float((actual['projected_scores'][2]-references[0]['scores']).abs().max())
    zero=joint_replay(p,receivers,actions,0,2,0.)
    errors['zero_rate_static']=float((zero['projected_scores']-zero['frozen_scores']).abs().max())
    suffix=tokens.clone();suffix[6:]=(suffix[6:]+1)%len(model.embed.weight)
    later=joint_replay(packet(model,suffix,eos_token=eos_token),receivers,actions,0,2,.01)
    errors['causal_first_loss']=float((actual['actual_scores'][:,0]-later['actual_scores'][:,0]).abs().max())
    identity=joint_replay(p,receivers,torch.zeros_like(actions),0,2,.01)
    assert identity['actual_worlds']==1;errors['no_op_identity']=float((identity['actual_scores'][0]-identity['actual_scores'][1]).abs().max())
    assert max(errors.values())<1e-10,errors
    return dict(vocabulary=len(model.embed.weight),depth=len(model.layers),targets=len(p.targets),errors=errors,actual_worlds=actual['actual_worlds'],readout_updates=actual['readout_updates'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();checks=[]
    for depth in [2,4,8]:
        torch.manual_seed(950+depth);model=KeyedTokenLM(8,3,depth,2,4,1,0.)
        model.embed=torch.nn.Embedding(23,8);model.bias=torch.nn.Parameter(torch.randn(23)*.02);model.requires_grad_(False);model.eval()
        for mode in ['ordinary','cross_eos','final_eos']:
            tokens=torch.tensor([2,4,6,8,4,2,6,8])
            if mode=='cross_eos':tokens[5]=7
            if mode=='final_eos':tokens[-1]=7
            checks.append(dict(mode=mode,**check(model,tokens,7)))
    # Extreme normalizers use stable exact exclusion, including near-unit receiver mass.
    extreme=[]
    for logits in [torch.tensor([100.,-100.,-80.]),torch.tensor([-120.,20.,-70.])]:
        for beta in [-160.,0.,160.]:
            receivers=[0,1,2];bonus=torch.full((3,),beta);pred=projected_costs(logits,receivers,bonus,2)
            true=torch.stack([F.cross_entropy((logits+F.one_hot(torch.tensor(r),3)*beta)[None],torch.tensor([2])) for r in receivers])
            error=float((pred-true).abs().max());assert error<1e-10;extreme.append(error)
    source=json.loads((ROOT/SOURCE).read_text());config=source['args'];checkpoint=ROOT/source['checkpoint'];sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model=KeyedTokenLM(config['d'],config['modes'],config['layers'],config['heads'],config['dk'],config['keyed'],config['dropout']);model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'));model.requires_grad_(False);model.eval()
    data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');offset=config['train_tokens']+4096
    tokens=torch.tensor(np.array(data[offset:offset+8],dtype=np.int64));checks.append(dict(mode='proper_gpt2_train',**check(model,tokens,50256)))
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,contracts=checks,extreme_normalizer_errors=extreme,source_model_result=SOURCE,source_checkpoint_sha256=sha,train_token_offset=offset,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},scope='Native temporal/keyed carrier fixed, independent lexical fast vector and Adam states. Small23-vocabulary source-world parity at depths2/4/8 with ordinary/cross-EOS/final-EOS paths; actual pretrained50257 GPT-2 vocabulary two-layer source, fresh TRAIN suffix. Full normalized forecasts, readout gradients/final vectors/Adam, projected reference, zero-step static, suffix causality, no-op and extreme logits contracted. Context four tokens, four forecast/readout updates; resets carried numerical/key/readout state on observed EOS. No policy fit/public validation/TEST, persistent-across-credit-window or all-carrier/depth-quality claim. Full source pretraining inherited, all independent contract worlds/readout updates paid inside whole wall.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
