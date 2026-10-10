"""B1/R1 independent Adam worlds for native optional route credit."""
import argparse,hashlib,json,resource,time
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
import unseen_receiver_credit as native
ROOT=native.ROOT
class BranchReadout(nn.Module):
    def __init__(self,linear,worlds):
        super().__init__();self.weight=nn.Parameter(linear.weight.detach()[None].repeat(worlds,1,1));self.bias=nn.Parameter(linear.bias.detach()[None].repeat(worlds,1))
    def forward(self,h):return torch.einsum('btd,bod->bto',h.detach(),self.weight)+self.bias[:,None]
def replay(source,t,m,receivers,actions,lr):
    routes=torch.tensor(receivers).repeat_interleave(len(actions));delta=actions.repeat(len(receivers),1);worlds=len(routes);model=native.actor(len(source.layers),0);model.load_state_dict(source.state_dict());model.requires_grad_(False);model.mark_ctx=BranchReadout(model.mark_ctx,worlds);opt=torch.optim.Adam(model.mark_ctx.parameters(),lr=lr);tt=t.repeat(worlds,1);mm=m.repeat(worlds,1);scores=[];versions=[]
    for j in range(4,m.shape[1]):
        opt.zero_grad();loss,_=native.losses(model,tt[:,:j+1],mm[:,:j+1],3,routes,delta);current=loss[:,-1];assert torch.isfinite(current).all();scores.append(current.detach());versions.append(dict(target=j,issued_version=j-4,feedback_update_version=j-3));current.sum().backward();opt.step()
    return torch.stack(scores,1),{k:v.detach().clone() for k,v in model.mark_ctx.state_dict().items()},versions
def contracts():
    source=native.actor(4,170);t,m=native.base.batch(456789,size=1,pairs=2);actions=native.base.actions()[:3];routes=[4,7]
    with torch.no_grad():fixed=native.responses(source,t,m,3,routes,actions)[0]
    zero,_,_=replay(source,t,m,routes,actions,0.);z=zero.mean(1).reshape(2,3);zero_error=float((z-z[:,:1]-fixed).abs().max());assert zero_error<1e-11
    batch,head,versions=replay(source,t,m,routes,actions,.01);score_error=state_error=0.
    for i,r in enumerate(routes):
        for k,action in enumerate(actions):
            solo,sh,_=replay(source,t,m,[r],action[None],.01);w=i*len(actions)+k;score_error=max(score_error,float((solo[0]-batch[w]).abs().max()));state_error=max(state_error,max(float((sh[key][0]-head[key][w]).abs().max()) for key in sh))
    assert score_error<1e-11 and state_error<1e-11
    noop_error=float((batch[0]-batch[len(actions)]).abs().max());assert noop_error==0
    changed=m.clone();changed[:,5:]=(changed[:,5:]+1)%8;other,_,_=replay(source,t,changed,routes,actions,.01);causal_error=float((batch[:,0]-other[:,0]).abs().max());assert causal_error==0
    adapted=batch.mean(1).reshape(2,3);difference=float(((adapted-adapted[:,:1])-fixed).abs().max())
    return dict(zero_lr_fixed_response_error=zero_error,batched_independent_score_error=score_error,batched_independent_final_weight_error=state_error,no_op_world_error=noop_error,future_suffix_first_prediction_error=causal_error,adaptation_changes_intervention_response=difference,score_update_versions=versions,readout_parameters_per_world=sum(v[0].numel() for v in head.values()),scope='Native frozen temporal/keyed carrier with separate adapting mark readouts and Adam coordinates. Causal full-prefix replay, not complete deep actor learning or sparse asynchronous execution. Fixed response algebra exact only at zero adaptation.')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();start=time.monotonic()
    with FlopCounterMode(display=False) as counter:metrics=contracts()
    sources=['experiments/credit/optimizer_aware_route_credit.py',*native.SOURCES];out.write_text(json.dumps(dict(status='completed',battle='B1/R1',tag=a.tag,metrics=metrics,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,supported_flops=counter.get_total_flops(),source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}),indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
