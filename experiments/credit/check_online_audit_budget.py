"""B1 selected-route budget contract; no native fit or TEST."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from online_audit_budget import AuditBudget
from predictable_audit_allocation_v2 import allocate
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
 pi=np.array([.1,.2,.3,.4]);v=np.array([.01,.25,4.,16.]);c=np.array([1.,2.,3.,4.]);B=1.2
 oracle=allocate(pi,v,c,B);price=float(np.mean(v/(c*oracle**2)));controller=AuditBudget(B,log_price=math.log(price))
 local=np.array([controller.issue(risk,cost).propensity for risk,cost in zip(v,c)])
 assert np.max(abs(local-oracle))<1e-14
 mean_dual=sum(float(pi[i])*(local[i]*(c[i]-B)+(1-local[i])*(-B)) for i in range(4));assert abs(mean_dual)<1e-14
 high=AuditBudget(B,log_price=math.log(price)+1);low=AuditBudget(B,log_price=math.log(price)-1)
 high_work=float(pi@(c*np.array([high.issue(r,k).propensity for r,k in zip(v,c)])))
 low_work=float(pi@(c*np.array([low.issue(r,k).propensity for r,k in zip(v,c)])))
 assert high_work<B<low_work
 # No enumeration is required by the controller's issue/consume interface.
 packet=controller.issue(1e300,1e-300);assert packet.propensity==1
 controller.consume(packet,audited=True,step=.001)
 zero=controller.issue(0,1);assert abs(zero.propensity-.02)<1e-15
 out=dict(status='completed',battle='B1/R1 selected-route audit budget',tag=a.tag,oracle_price=price,local_oracle_probability_error=float(np.max(abs(local-oracle))),conditional_dual_mean_at_budget=mean_dual,underpriced_expected_work=low_work,overpriced_expected_work=high_work,controller_state_scalars=4,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ['experiments/credit/check_online_audit_budget.py','experiments/credit/online_audit_budget.py','experiments/credit/predictable_audit_allocation_v2.py']},scope='Finite oracle-risk contract: selected-route local rates equal global constrained optimum. Dual drift has corrective sign; no convergence proof under learned/nonstationary risks, native calibration, hard event budget, quality or full sparse-learning win.')
 with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
