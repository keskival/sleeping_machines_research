"""B1/R1 fixed-point credit numerical/work contract; no model fitting or TEST."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from fixed_point_credit import solve_credit
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
 rng=np.random.default_rng(913);n=32;M=rng.normal(size=(n,n));M*=.45/np.linalg.norm(M,2)
 def transport(x,*,context,version):
  if context!='contract' or version!=7:raise ValueError('stale operator')
  return M.T@x
 b=rng.normal(size=n);tick=time.monotonic();cold=solve_credit(b,transport,context='contract',version=7,contraction=.45,tolerance=1e-10);seconds=time.monotonic()-tick
 exact=np.linalg.solve(np.eye(n)-M.T,b);err=float(np.linalg.norm(cold.vector-exact));assert cold.converged and err<=cold.error_bound+1e-14
 next_b=b+1e-4*rng.normal(size=n)
 warm=solve_credit(next_b,transport,context='contract',version=7,contraction=.45,tolerance=1e-10,initial=cold.vector)
 fresh=solve_credit(next_b,transport,context='contract',version=7,contraction=.45,tolerance=1e-10)
 assert warm.converged and np.linalg.norm(warm.vector-fresh.vector)<2e-10 and warm.operator_calls<fresh.operator_calls
 short=solve_credit(b,transport,context='contract',version=7,contraction=.45,max_calls=2);assert not short.converged
 try:solve_credit(b,transport,context='contract',version=8,contraction=.45)
 except ValueError:pass
 else:raise AssertionError('stale version accepted')
 out=dict(status='completed',battle='B1/R1 reciprocal fixed-point credit',tag=a.tag,dimension=n,cold_calls=cold.operator_calls,warm_calls=warm.operator_calls,fresh_calls=fresh.operator_calls,exact_error=err,residual_norm=cold.residual_norm,certified_error_bound=cold.error_bound,cold_solve_s=seconds,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ['experiments/credit/fixed_point_credit.py','experiments/credit/check_fixed_point_credit.py']},scope='Frozen synthetic linear contraction; bound supplied from measured matrix norm. Transport callback hides no charged work: one dense matvec per call here. No native actor/critic integration, coupled learning convergence, quality or end-to-end efficiency claim.')
 with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
