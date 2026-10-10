"""R1/B1 learned-response cotangent, version and conditional-emission contracts."""
import argparse,hashlib,json,time
import torch
from structured_future_credit_v5 import KernelCritic,base,bound,SOURCES as BASE_SOURCES
from response_credit_policy import activations,cotangent,propose
SOURCES=['experiments/credit/check_response_policy.py','experiments/credit/response_credit_policy.py',*BASE_SOURCES]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();rows=[]
    for depth in (2,4,8):
        model=base.actor(depth,181);t,m=base.batch(181,size=1,pairs=2);x=base.prefix_packet(model,t,m,3);critic=KernelCritic(model.M,bound(model)).double();packet=activations(critic,x,'actor0','credit0')
        g=cotangent(packet);action=torch.zeros(1,1,4,requires_grad=True);exact=torch.autograd.grad(critic.values(x,action).sum(),action)[0].reshape(1,4);error=float((g-exact).abs().max());assert error<1e-12
        delta,meta=propose(packet,'actor0','credit0');assert delta.norm()<=.25+1e-12 and meta['status']=='bounded_uncertified'
        stale,status=propose(packet,'actor1','credit0');assert not stale.any() and status['status']=='stale_recompute_required'
        certified,info=propose(packet,'actor0','credit0',gradient_error=0.)
        original=float(critic.values(x,torch.zeros_like(action)));changed=float(critic.values(x,certified[:,None]));upper=info['future_loss_upper_bound'][0];assert changed-original<=upper+1e-12
        withheld,info=propose(packet,'actor0','credit0',gradient_error=critic.U);assert not withheld.any()
        rows.append(dict(depth=depth,cotangent_error=error,proposal_norm=float(delta.norm()),retained_activation_elements=sum(v.numel() for v in packet.values() if isinstance(v,torch.Tensor)),status='passed',scope='Conditional certificate tested with surrogate itself as target, hence zero gradient error; not a certificate for actual future data.'))
    result=dict(status='completed',tag=a.tag,battle='R1/B1',metrics=rows,wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((base.ROOT/p).read_bytes()).hexdigest() for p in SOURCES});output=base.ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
