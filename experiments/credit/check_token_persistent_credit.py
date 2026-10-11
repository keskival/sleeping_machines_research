"""R1 carried factual carrier/readout/Adam parity versus full native episode."""
import argparse,hashlib,json,math,resource,time
from copy import deepcopy
import torch
from torch.nn import functional as F
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,load,np
from token_forward_credit import packet
from token_episode_state import consume
from token_persistent_credit import joint_replay_carry
SOURCES=['experiments/credit/check_token_persistent_credit.py','experiments/credit/token_persistent_credit.py','experiments/credit/token_forward_credit.py','experiments/credit/token_episode_state.py','experiments/r1_token_keyed_lm.py','experiments/tpp/race_tpp_v5.py']

def full_native(model,tokens,schedule,eos):
    state=None;captured=[];hook=model.query_w.register_forward_hook(lambda module,args,result:captured.append(result.detach()))
    def learner():
        z=torch.nn.Parameter(model.embed.weight.new_zeros(model.embed.embedding_dim));return z,torch.optim.Adam([z],lr=.01)
    z,opt=learner();scores=[]
    try:
        for j in range(len(tokens)-1):
            if int(tokens[j])==eos:state=None;z,opt=learner()
            with torch.no_grad():
                forecast,state=consume(model,tokens[j:j+1],state,parameter_version=0)
                if j in schedule:
                    r,delta=schedule[j];keys=dict(state['keys']);now=state['position']-1
                    value=torch.zeros_like(delta) if r not in keys else keys[r][0]*torch.exp(-model.log_rate.exp()*(now-keys[r][1]))[:,None]
                    keys[r]=(value+delta,now);state=dict(state,keys=keys)
                    beta=(captured[-1].reshape(model.heads,model.dk)*delta).sum()/math.sqrt(model.dk)
                    lp=(forecast['log_probabilities']+F.one_hot(torch.tensor(r),len(model.embed.weight))*beta).log_softmax(-1)
                    forecast=dict(forecast,log_probabilities=lp)
            if j>=3:
                opt.zero_grad();logits=forecast['log_probabilities'][0]+model.embed.weight.detach()@z
                loss=F.cross_entropy(logits[None],tokens[j+1:j+2]);scores.append(loss.detach());loss.backward();opt.step()
    finally:hook.remove()
    if int(tokens[-1])==eos:state=None;z,opt=learner()
    return torch.stack(scores),z.detach(),opt.state_dict(),state


def check(model,tokens,eos):
    carrier=learner=None;scores=[];schedule={};snapshot_checks=0
    for step,start in enumerate([0,11,19]):
        index=3 if step==0 else 0;end=12 if step==0 else start+9
        current=tokens[start:end];actions=torch.zeros(2,model.heads,model.dk)
        actions[1,step%model.heads,step%model.dk]=.25 if step%2==0 else -.25
        receiver=[int(current[index]),len(model.embed.weight)-1 if step==1 else int(tokens[2])]
        schedule[start+index]=(receiver[1],actions[1])
        before=deepcopy(learner);p=packet(model,current,index=index,initial_state=carrier,eos_token=eos)
        actual=joint_replay_carry(p,receiver,actions,1,0,.01,initial_learner=learner,diagnostics=True)
        if before is not None:
            assert torch.equal(before['z'],learner['z']);snapshot_checks+=1
            for key,fields in before['optimizer']['state'].items():
                for field,value in fields.items():
                    if isinstance(value,torch.Tensor):assert torch.equal(value,learner['optimizer']['state'][key][field]);snapshot_checks+=1
        scores.append(actual['actual_scores'][0]);carrier=actual['factual_carrier'];learner=actual['factual_learner']
    truth,z,opt,state=full_native(model,tokens,schedule,eos)
    errors=dict(scores=float((torch.cat(scores)-truth).abs().max()),fast_state=float((learner['z']-z).abs().max()))
    assert learner['optimizer']['state'].keys()==opt['state'].keys()
    errors['adam']=max([float((learner['optimizer']['state'][k][n]-v[n]).abs().max()) for k,v in opt['state'].items() for n in ['step','exp_avg','exp_avg_sq']]+[0.])
    if state is None:assert carrier is None
    else:
        assert carrier['position']==state['position'] and carrier['keys'].keys()==state['keys'].keys()
        errors['layers']=max(float((a-b).abs().max()) for a,b in zip(carrier['layers'],state['layers']))
        errors['keys']=max(float((carrier['keys'][k][0]*torch.exp(-model.log_rate.exp()*(carrier['position']-carrier['keys'][k][1]))[:,None]-v[0]*torch.exp(-model.log_rate.exp()*(state['position']-v[1]))[:,None]).abs().max()) for k,v in state['keys'].items())
    assert max(errors.values())<1e-10,errors
    return dict(vocabulary=len(model.embed.weight),depth=len(model.layers),errors=errors,incoming_snapshot_checks=snapshot_checks,targets=len(truth),credit_windows=3)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();checks=[]
    for depth in [2,4,8]:
        torch.manual_seed(1030+depth);model=KeyedTokenLM(8,3,depth,2,4,1,0.);model.embed=torch.nn.Embedding(23,8);model.bias=torch.nn.Parameter(torch.randn(23)*.02);model.requires_grad_(False);model.eval()
        for reset in [False,True]:
            tokens=torch.tensor(([2,4,6,8]*7)[:28])
            if reset:tokens[14]=7;tokens[-1]=7
            checks.append(dict(reset=reset,**check(model,tokens,7)))
    source=json.loads((ROOT/SOURCE).read_text());c=source['args'];model=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);checkpoint=ROOT/source['checkpoint'];model.load_state_dict(torch.load(checkpoint,weights_only=True));model.requires_grad_(False);model.eval()
    data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');offset=c['train_tokens']+12288;tokens=torch.tensor(np.array(data[offset:offset+28],dtype=np.int64));checks.append(dict(reset='observed_gpt2',**check(model,tokens,50256)))
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,contracts=checks,source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},scope='Three8-target credit windows with3 selected hard writes, same continuous native state and independent actual readout/Adam carry. Full source episode scores/final fast state/optimizer/layers/lazy keys agree; incoming snapshots unmodified. Small vocab depth2/4/8 with middle/final EOS and actual pretrained50257 vocabulary2layer fresh TRAIN. No actor fitting/TEST/depth-quality or continuous full-carrier-gradient claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
