"""B1 native changing-layer local continuation-credit contract.
The resolvent is a declared local linear continuation target, not the existing
whole-model loss gradient or proof of equilibrium learning convergence.
"""
import argparse,hashlib,json,platform,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tpp'))
from race_tpp_v19 import TemporalMemoryLayer
from reciprocal_transport import DirectionalPacket
from reciprocal_transport_projection import ProjectedTransport as ReciprocalTransport
from fixed_point_credit import solve_credit

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args()
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(916)
 layer=TemporalMemoryLayer(8,4,0.).eval();opt=torch.optim.Adam(layer.parameters(),lr=.001)
 x=torch.randn(16);dt=torch.tensor([[0.,.37]]);prior=None;history=[];start=time.monotonic()
 for version in range(4):
  def forward(z):return layer(z.reshape(1,2,8),dt).flatten()
  tick=time.monotonic();transport=ReciprocalTransport(torch.zeros(16,16),context='native-cut',version=version)
  for direction in torch.eye(16):
   _,response=torch.func.jvp(forward,(x,),(direction,))
   transport.teach_packet(DirectionalPacket.capture('native-cut',version,direction,response))
  teaching_s=time.monotonic()-tick
  B=transport.matrix.detach().numpy();tick=time.monotonic();norm=float(np.linalg.norm(B,2));gamma=.4/max(norm,1e-12);bound_s=time.monotonic()-tick
  xx=x.detach().requires_grad_();yy=forward(xx);b=yy.detach().numpy().copy();context='native-cut'
  # Current forward graph and parameters remain unchanged during both solves.
  def exact_operator(vector,*,context,version: int):
   if context!='native-cut' or version!=issued_version:raise ValueError('stale native solve')
   return gamma*torch.autograd.grad(yy,xx,torch.tensor(vector),retain_graph=True)[0].detach().numpy()
  def taught_operator(vector,*,context,version):
   return gamma*transport.predict(torch.tensor(vector),context=context,version=version).numpy()
  issued_version=version
  tick=time.monotonic();exact=solve_credit(b,exact_operator,context=context,version=version,contraction=.4,tolerance=1e-9);exact_s=time.monotonic()-tick
  tick=time.monotonic();taught=solve_credit(b,taught_operator,context=context,version=version,contraction=.4,tolerance=1e-9);taught_s=time.monotonic()-tick
  tick=time.monotonic();warm=solve_credit(b,taught_operator,context=context,version=version,contraction=.4,tolerance=1e-9,initial=prior);warm_s=time.monotonic()-tick
  direct=np.linalg.solve(np.eye(16)-gamma*B,b)
  error=float(np.linalg.norm(taught.vector-exact.vector));direct_error=float(np.linalg.norm(taught.vector-direct));assert exact.converged and taught.converged and warm.converged and error<1e-9 and direct_error<=taught.error_bound+1e-12
  assert np.linalg.norm(warm.vector-direct)<=warm.error_bound+1e-12
  # Standard observed loss updates native parameters; solves are diagnostics.
  opt.zero_grad();yy.square().mean().backward();opt.step();prior=warm.vector.copy()
  history.append(dict(version=version,operator_norm=norm,gamma=gamma,contraction_bound=.4,forward_probes=16,transport_floats=256,teaching_s=teaching_s,bound_s=bound_s,exact_solve_s=exact_s,taught_solve_s=taught_s,warm_solve_s=warm_s,exact_calls=exact.operator_calls,taught_calls=taught.operator_calls,warm_calls=warm.operator_calls,taught_exact_error=error,direct_error=direct_error,residual_error_bound=taught.error_bound))
 sources=['experiments/credit/native_reciprocal_fixed_point_v2.py','experiments/credit/fixed_point_credit.py','experiments/credit/reciprocal_transport.py','experiments/credit/reciprocal_transport_projection.py','experiments/tpp/race_tpp_v19.py']
 result=dict(status='completed',tag=args.tag,battle='B1/R1 reciprocal native local credit',history=history,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,hardware=dict(host=platform.node(),threads=1,dtype='float64'),scope='Two-event native temporal layer, four changing parameter versions. Declared frozen local linear continuation lambda=b+gamma*J.T*lambda; adaptive gamma set from exhaustive current dense transport norm to bound contraction0.4. Full probes/norm/exact reverse audits charged. Standard layer loss updates weights; resolvent is not substituted as its gradient. No whole-family objective, useful learned generalization, sparse-cost, task quality, TEST or end-to-end efficiency claim.')
 with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
