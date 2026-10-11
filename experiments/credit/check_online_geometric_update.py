"""R1 exact next-loss derivative through issued geometric update and reset/version."""
import argparse,hashlib,json,math,time
import torch
from torch.nn import functional as F
from check_token_forward_credit import ROOT
from online_geometric_update import geometric_step,UpdateFunction,OnlineUpdateLearner

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);checks=[]
    for magnitude in [.01,10.]:
        torch.manual_seed(99715);embedding=torch.randn(23,8);base=torch.randn(23);y=torch.tensor([7]);z=torch.randn(8)*.01;moment=torch.randn(8)*.001;product=torch.tensor(.9);g=torch.randn(8)*magnitude;u=torch.randn(8,3);h=u@u.T+torch.eye(8)*.3
        def issued(raw):
            eta=.01*torch.exp(math.log(10)*raw[0].tanh());damping=torch.exp(math.log(10)*raw[1].tanh());beta=.99*torch.sigmoid(raw[2]+math.log(10));return geometric_step(z,moment,product,g,h,eta,damping,beta)[0]
        raw=torch.tensor([.2,-.1,.3],requires_grad=True);next_z=issued(raw);logits=base+embedding@next_z;loss=F.cross_entropy(logits[None],y)
        exact=torch.autograd.grad(loss,raw,retain_graph=True)[0];future_gradient=embedding.T@logits.detach().softmax(0)-embedding[int(y)]
        observed=torch.autograd.grad(next_z,raw,future_gradient)[0];finite=[]
        for k in range(3):
            plus=raw.detach().clone();minus=plus.clone();plus[k]+=1e-6;minus[k]-=1e-6
            finite.append((F.cross_entropy((base+embedding@issued(plus))[None],y)-F.cross_entropy((base+embedding@issued(minus))[None],y))/(2e-6))
        errors=dict(analytic=float((exact-observed).abs().max()),finite_difference=float((exact-torch.stack(finite)).abs().max()));assert max(errors.values())<1e-7;assert float((next_z-z).norm())<=.08000000001;checks.append(dict(magnitude=magnitude,errors=errors))
    function=UpdateFunction(42);learner=OnlineUpdateLearner(function,8,True);g=torch.randn(8);h=torch.eye(8);hidden=torch.randn(8);learner.feedback(g,h,hidden,2.,0);assert learner.meta_updates==0
    learner.feedback(g*.4,h,hidden,2.,1);assert learner.meta_updates==1;learner.reset();assert learner.pending is None and not bool(learner.z.any());assert learner.meta_updates==1
    learner.feedback(g,h,hidden,2.,2);learner.version+=1
    try:learner.feedback(g,h,hidden,2.,3)
    except ValueError:pass
    else:raise AssertionError('Stale credit graph accepted')
    paths=['experiments/credit/check_online_geometric_update.py','experiments/credit/online_geometric_update.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,checks=checks,wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='One-step futureCE meta-gradient, active/inactive normcap, retainedgraph/update/reset/version; synthetic23class8dimensional readout, no quality/depthclaim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
