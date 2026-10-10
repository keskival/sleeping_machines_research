"""B1/R1 finite numerical witnesses for information, audits, depth and versions."""
import argparse,hashlib,itertools,json,resource,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
NOTE='experiments/theory/CREDIT_INFORMATION_AND_DEPTH_ECONOMICS.md'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();start=time.monotonic()
    m=np.array([[1.,2.],[-1.,0.],[3.,-2.],[1.,4.]])
    noise=np.array([[-.3,.2],[.3,-.2]])
    y=m[:,None,:]+noise[None,:,:];zmean=m.reshape(2,2,2).mean(1).repeat(2,axis=0);h=zmean+np.array([.1,-.2])
    norm=lambda x:float(np.square(x).sum(-1).mean())
    innovation=norm(y-m[:,None,:]);compression=norm(m-zmean);fitting=norm(zmean-h);risk=norm(y-h[:,None,:]);information_error=abs(risk-innovation-compression-fitting)
    coarse_risk=norm(y-zmean[:,None,:]);fine_risk=innovation;gain=norm(m-zmean);nested_error=abs(coarse_risk-fine_risk-gain)
    assert information_error<1e-12 and nested_error<1e-12
    r=np.array([.01,.02,.03]);c=np.array([1.,2.,4.]);tau=1.;S=np.sqrt(r*c).sum();aa=np.sqrt(r/c)*S/(tau+r.sum());assert np.all((aa>0)&(aa<1));cost=float(aa@c);variance=float(((1/aa-1)*r).sum());optimal_error=abs(cost-S*S/(tau+r.sum()));constraint_error=abs(variance-tau)
    # Compare every feasible point on a grid to the analytic interior optimum.
    best_grid=float('inf');feasible=0
    for cand in itertools.product(np.linspace(.01,1.,20),repeat=3):
        b=np.array(cand)
        if float(((1/b-1)*r).sum())<=tau:best_grid=min(best_grid,float(b@c));feasible+=1
    assert feasible and best_grid>=cost-1e-12
    target=np.sqrt(r);mean=np.zeros(3);audit_variance=0.
    for bits in itertools.product([0,1],repeat=3):
        b=np.array(bits);prob=float(np.prod(np.where(b,aa,1-aa)));estimate=b*target/aa;mean+=prob*estimate;audit_variance+=prob*float(np.square(estimate-target).sum())
    mean_error=float(np.max(np.abs(mean-target)));variance_error=abs(audit_variance-tau);assert max(optimal_error,constraint_error,mean_error,variance_error)<1e-12
    # Same target, stale predictor: only the audit label's version controls mean.
    current=np.array([2.,-1.]);old=np.array([1.,1.]);stale=np.array([.5,-.3]);audit=.2
    corrected=(1-audit)*stale+audit*(stale+(current-stale)/audit)
    wrong=(1-audit)*stale+audit*(stale+(old-stale)/audit)
    current_error=float(np.max(np.abs(corrected-current)));old_bias=float(np.linalg.norm(wrong-current));assert current_error<1e-12 and old_bias>1.
    rf=norm(y-zmean[:,None,:]);rs=norm(y-h[:,None,:]);drift=np.sqrt(norm(h-zmean));stale_upper=(np.sqrt(rf)+drift)**2;assert rs<=stale_upper+1e-12
    # Depth witnesses share one unnormalized added-MSE tolerance and cost units.
    depths=[1,4,16,64];floor_ratios=[D*.01/(tau+D*.01) for D in depths];improving_ratios=[(.01/D)/(tau+.01/D) for D in depths]
    assert np.all(np.diff(floor_ratios)>0) and np.all(np.diff(improving_ratios)<0)
    # Exact quadratic: the descent lemma equals the computed expected loss.
    theta=np.array([1.,-2.]);eta=.1;g=theta;pred=np.zeros(2);audit=.25
    expected=sum(prob*.5*np.square(theta-eta*est).sum() for prob,est in [(1-audit,pred),(audit,pred+(g-pred)/audit)])
    added=(1/audit-1)*float(g@g);bound=.5*float(theta@theta)-eta*float(g@g)+.5*eta**2*(float(g@g)+added);descent_error=abs(expected-bound);assert descent_error<1e-12
    sources=['experiments/credit/check_credit_information_economics.py',NOTE]
    result=dict(status='completed',battle='B1/R1',tag=a.tag,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,metrics=dict(information_decomposition_error=information_error,nested_information_error=nested_error,risk_parts=dict(innovation=innovation,compression=compression,fitting=fitting,total=risk),optimal_audit_probabilities=aa.tolist(),optimal_audit_cost=cost,feasible_grid_points=feasible,best_grid_cost=best_grid,optimal_cost_identity_error=optimal_error,budget_identity_error=constraint_error,enumerated_audit_outcomes=8,audit_mean_error=mean_error,audit_variance_error=variance_error,current_label_stale_prediction_mean_error=current_error,old_label_current_target_bias=old_bias,stale_risk=rs,stale_risk_upper=stale_upper,depths=depths,fixed_residual_audit_ratios=floor_ratios,improving_residual_audit_ratios=improving_ratios,quadratic_descent_identity_error=descent_error),source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},scope='Finite arithmetic witnesses of proved conditional identities and an interior audit optimum; no neural fitting, general formal proof checking, real-stream depth efficiency or hardware claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['metrics'],indent=2))
if __name__=='__main__':main()
