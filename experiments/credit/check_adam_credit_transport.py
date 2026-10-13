"""R1 future-credit integration gate against actual fixed-version Adam proposal derivatives."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sleeping_machines.learning.adam_credit_transport import adam_gradient_cotangent

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);rows=[]
 for size in (4,17):
  for amplitude in (.01,3.):
   for step in (1,13):
    g=torch.linspace(-.9,1.1,size,dtype=torch.float64)*amplitude;g.requires_grad_()
    m=torch.linspace(-.2,.3,size,dtype=torch.float64);v=torch.linspace(.1,.8,size,dtype=torch.float64)
    lam=torch.linspace(-.7,.2,size,dtype=torch.float64);nu=torch.linspace(.3,-.1,size,dtype=torch.float64);xi=torch.linspace(-.2,.6,size,dtype=torch.float64)
    clipped=g*(1/(g.norm()+1e-6)).clamp(max=1);mn=.9*m+.1*clipped;vn=.999*v+.001*clipped.square()
    proposal=-.003*(mn/(1-.9**step))/((vn/(1-.999**step)).sqrt()+1e-8)
    exact=torch.autograd.grad((proposal*lam+mn*nu+vn*xi).sum(),g)[0]
    predicted=adam_gradient_cotangent(g.detach(),m,v,lam,step=step,lr=.003,moment1_cotangent=nu,moment2_cotangent=xi)
    error=float((predicted-exact).abs().max());assert error<1e-13
    rows.append(dict(size=size,amplitude=amplitude,step=step,clipped=bool(g.norm()+1e-6>1),maximum_error=error))
 # Continuous composite derivative at a never-updated coordinate.
 z=torch.zeros(3,dtype=torch.float64);lam=torch.ones_like(z)
 predicted=adam_gradient_cotangent(z,z,z,lam,step=1,lr=.003)
 assert torch.allclose(predicted,torch.full_like(z,-.003/1e-8),rtol=1e-14,atol=1e-9)
 names=[str(Path(__file__).relative_to(ROOT)),'sleeping_machines/learning/adam_credit_transport.py','sleeping_machines/__init__.py'];sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
 d=dict(status='completed',battle='R1',gate='PASS',rows=rows,untouched_coordinate_limit=True,source_sha256=sources,scope='Float64 fixed-version clip/AdamW gradient cotangent, with parameter and first/second-moment future demands. Dense autograd proposal audit includes clipping on/off and steps1/13; weight decay has zero derivative wrt gradient at fixed parameter. No fitted critic, forecast-quality or whole-work claim. Global clip needs a complete-vector contraction; subsequent optimizer/state paths require their own cotangents.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
 print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
