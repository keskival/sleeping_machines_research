"""B1/R1 finite conditional-gradient/propensity-allocation contract, no fit."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic()
    soft=np.array([.2,.3,.5]);pi=.9*soft+.1/3
    jac=.9*(np.diag(soft)-np.outer(soft,soft));score=jac/pi[:,None]
    outcomes=np.array([[-.03,.01],[.02,.04],[-.01,.05]])
    future=np.array([.4,.6]);q=outcomes@future;h=np.array([.01,-.02,.03])
    g=(pi*q)@score;gh=(pi*h)@score
    errors=[]
    for p in [np.ones(3),np.full(3,.25),np.array([.125,.5,.8])]:
        mean=np.zeros(3);second=0.
        for route in range(3):
            for j in range(2):
                for audit,mass in [(0,1-p[route]),(1,p[route])]:
                    gradient=gh+audit/p[route]*(outcomes[route,j]-h[route])*score[route]
                    probability=pi[route]*future[j]*mass
                    mean+=probability*gradient;second+=probability*gradient@gradient
        v=((outcomes-h[:,None])**2@future)*(score**2).sum(1)
        variance=second-mean@mean;formula=pi@(v/p)-(g-gh)@(g-gh)
        noise=((outcomes-q[:,None])**2@future)*(score**2).sum(1)
        residual=(q-h)[:,None]*score
        between=pi@((residual**2).sum(1))-(pi@residual)@(pi@residual)
        decomposition=pi@(noise/p)+between+pi@((1/p-1)*(q-h)**2*(score**2).sum(1))
        error=max(float(abs(mean-g).max()),float(abs(variance-formula)),float(abs(variance-decomposition)))
        assert error<1e-14;errors.append(error)
    # Perfect conditional-mean credit leaves irreducible future variance.
    p=np.full(3,.25);gh=(pi*q)@score
    noise=((outcomes-q[:,None])**2@future)*(score**2).sum(1)
    exact_noise=pi@(noise/p);assert exact_noise>0
    # Componentwise KKT witness, including lower/upper bound clipping.
    v=np.array([.0001,.01,1.]);cost=np.array([1.,3.,2.]);lam=.1;pmin=.08
    allocated=np.clip(np.sqrt(lam*v/cost),pmin,1.)
    rng=np.random.default_rng(23);alternatives=rng.uniform(pmin,1.,(10000,3))
    optimum=pi@(cost*allocated+lam*v/allocated)
    other=(cost*alternatives+lam*v/alternatives)@pi
    assert float(other.min())>=optimum-1e-14
    # At the allocation's variance budget it is minimum cost; Lagrange bound
    # witnesses this without treating a finite random search as a proof.
    variance_budget=pi@(v/allocated);allocated_cost=pi@(cost*allocated)
    feasible=(v/alternatives)@pi<=variance_budget+1e-14
    if feasible.any():assert float((alternatives@ (pi*cost))[feasible].min())>=allocated_cost-1e-14
    # A low-loss route need not be the low-variance teaching reference.
    worlds=np.array([[-1.,1.],[2.99,3.01],[4.01,3.99]])
    mass=np.full(2,.5);levels=worlds@mass;p=np.full(3,.25)
    residual=worlds-levels[:,None];weights=pi*(score**2).sum(1)/p
    center=(weights@residual)/weights.sum();constant=(weights[:,None]*(residual-center)**2)@mass
    references=[];medoid_errors=[];expected=(pi*levels)@score
    for b in range(3):
        risk=float(weights@((residual-residual[b])**2@mass))
        medoid=float(constant.sum()+weights.sum()*((residual[b]-center)**2@mass))
        mean=np.zeros(3);second=0.
        for route in range(3):
            for j in range(2):
                for audit,coin in [(0,1-p[route]),(1,p[route])]:
                    grad=expected+audit/p[route]*(residual[route,j]-residual[b,j])*score[route]
                    prob=pi[route]*mass[j]*coin;mean+=prob*grad;second+=prob*grad@grad
        error=max(abs(risk-medoid),float(abs(mean-expected).max()),abs(second-mean@mean-risk))
        assert error<1e-12;medoid_errors.append(error);references.append(risk)
    assert int(np.argmin(references))!=int(np.argmin(levels))
    metric=dict(enumerated_mean_variance_errors=errors,perfect_critic_future_noise_trace=float(exact_noise),
                allocated_propensities=allocated.tolist(),allocated_incremental_cost=float(allocated_cost),
                second_moment_budget=float(variance_budget),lagrange_gap_to_random_candidates=float(other.min()-optimum),
                feasible_random_candidates=int(feasible.sum()),positive_support_floor=pmin,
                reference_geometry_errors=medoid_errors,reference_gradient_variances=references,
                lowest_mean_loss_route=int(np.argmin(levels)),lowest_variance_reference=int(np.argmin(references)))
    result=dict(status='completed',battle='B1/R1',tag=a.tag,metrics=metric,wall_s=time.monotonic()-start,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                scope='Finite three-route/two-future algebra contracts. Action-dependent positive audit support preserves raw expected gradient, exact trace covariance includes score sensitivity, and a clipped allocation minimizes its Lagrangian. No learned residual-risk calibration, optimizer-step, native-quality or hardware efficiency result.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
