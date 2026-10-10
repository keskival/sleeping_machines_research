"""B1/R1 saved-head value/VJP and optimizer checks, paired kernel work."""
import argparse,hashlib,json,resource,time
from copy import deepcopy
import torch
from torch.utils.flop_counter import FlopCounterMode
from fit_optimizer_aware_credit import Correction,ROOT,native
from cached_learning_effect import values

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();begin=time.monotonic();src=ROOT/'experiments/results/credit/curie_paired_learning_effect_20261011T0030Z_fit.json';d=json.loads(src.read_text());cp=ROOT/d['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==d['checkpoint_sha256'];s=torch.load(cp,map_location='cpu',weights_only=False);teacher=torch.load(ROOT/s['source_checkpoint'],map_location='cpu',weights_only=False);z=teacher['dev_tensors']['z'];actions=native.base.actions()[None,None].expand(len(z),8,-1,-1);head=Correction(s['scale']).double();head.load_state_dict(s['head']);head.net[-1].bias.data.fill_(.2)
    old=head.values(z,actions);new=values(head,z,actions);error=float((old-new).abs().max());assert error<1e-12;g1=torch.autograd.grad(old.square().sum(),tuple(head.parameters()));g2=torch.autograd.grad(new.square().sum(),tuple(head.parameters()));grad_error=max(float((x-y).abs().max()) for x,y in zip(g1,g2));assert grad_error<1e-12
    h1=deepcopy(head);h2=deepcopy(head);o1=torch.optim.AdamW(h1.parameters(),lr=.003,weight_decay=.001);o2=torch.optim.AdamW(h2.parameters(),lr=.003,weight_decay=.001);target=teacher['dev_tensors']['aware']-teacher['dev_tensors']['fixed']
    for step in range(16):
        for h,opt,fast in [(h1,o1,False),(h2,o2,True)]:
            opt.zero_grad();pred=values(h,z,actions) if fast else h.values(z,actions);loss=((pred-target)/s['scale']).square().mean();loss.backward();opt.step()
    parameter_error=max(float((x-y).abs().max()) for x,y in zip(h1.parameters(),h2.parameters()));assert parameter_error<1e-10,parameter_error
    work=[]
    for name,fast in [('repeated_context',False),('cached_context',True)]:
        h=deepcopy(head);tick=time.monotonic()
        with FlopCounterMode(display=False) as forward:
            for iteration in range(20):
                with torch.no_grad():pred=values(h,z,actions) if fast else h.values(z,actions)
        fw=time.monotonic()-tick;tick=time.monotonic()
        with FlopCounterMode(display=False) as fitting:
            pred=values(h,z,actions) if fast else h.values(z,actions);pred.square().mean().backward()
        work.append(dict(arm=name,forward_iterations=20,forward_supported_flops=forward.get_total_flops(),forward_instrumented_wall_s=fw,one_forward_backward_supported_flops=fitting.get_total_flops(),backward_instrumented_wall_s=time.monotonic()-tick))
    sources=['experiments/credit/check_cached_learning_effect.py','experiments/credit/cached_learning_effect.py','experiments/credit/fit_optimizer_aware_credit.py',*native.SOURCES]
    result=dict(status='completed',battle='B1/R1',tag=a.tag,metrics=dict(value_error=error,parameter_vjp_error=grad_error,adam16_step_parameter_error=parameter_error,nonzero_bias_decay_preserved=True,work=work,prefixes=len(z),receivers=8,actions=9),whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},scope='Same learned correction and packets, operator/value/gradient/short optimizer witnesses. No teacher/actor refit. Instrumented paired kernel work includes context projection; excludes common actor/baseline/discovery/prior fitting. No whole-fit or deployment speed claim.');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['metrics'],indent=2))
if __name__=='__main__':main()
