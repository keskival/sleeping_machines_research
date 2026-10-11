"""B1/R1 conditional future-credit identity and route-dependent-input rejection."""
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
    pi=np.array([.2,.3,.5]);score=np.eye(3)-pi[None]
    futures=np.array([.4,.6]);worlds=np.array([[-.03,.01],[.02,.04],[-.01,.05]])
    contrasts=worlds-worlds[1];q=contrasts@futures
    exact=(pi*q)@score;rows=[];conditional_means=(pi[:,None]*contrasts).T@score
    future_variance=float(futures@((conditional_means-exact)**2).sum(1))
    for name,h in [('prefix_mean',q[:,None].repeat(2,1)),('revealed_exact',contrasts),('revealed_imperfect',contrasts+np.array([[.001],[-.002],[.003]]))]:
        p=np.array([.125,.25,.5]);mean=np.zeros(3);second=0.;conditional=[]
        for u in range(2):
            gu=np.zeros(3);model=(pi*h[:,u])@score
            for route in range(3):
                for audit,coin in [(0,1-p[route]),(1,p[route])]:
                    gradient=model+audit/p[route]*(contrasts[route,u]-h[route,u])*score[route]
                    mass=pi[route]*coin;gu+=mass*gradient
                    mean+=futures[u]*mass*gradient;second+=futures[u]*mass*gradient@gradient
            conditional.append(float(abs(gu-conditional_means[u]).max()))
        error=float(abs(mean-exact).max());assert max(error,*conditional)<1e-14
        variance=float(second-mean@mean)
        if name=='revealed_exact':assert abs(variance-future_variance)<1e-14
        rows.append(dict(arm=name,expected_gradient_error=error,conditional_gradient_errors=conditional,trace_variance=variance))
    assert rows[1]['trace_variance']<rows[0]['trace_variance']
    # Chosen-route loss is not an exogenous common future feature.
    # pi=[.5,.5], scalar score=[-.5,.5], L=[0,2], reference0.
    # h_A(a)=L_A * [0,1] has model term L_A*.25, zero residual.
    actual_gradient=.5;invalid_gradient=.25;assert invalid_gradient!=actual_gradient
    d=dict(status='completed',battle='B1/R1',tag=a.tag,rows=rows,
        irreducible_future_gradient_variance=future_variance,
        exact_future_over_prefix_variance=rows[1]['trace_variance']/rows[0]['trace_variance'],
        route_dependent_input_counterexample=dict(true_gradient=actual_gradient,invalid_estimator_mean=invalid_gradient),
        wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='Finite three-route/two-future enumeration with independent raw future observations and positive action-dependent audits. Exact and imperfect future credit preserve raw gradient; exact future prediction leaves future-gradient variance. Explicit chosen-route-loss counterexample rejects treating endogenous feedback as shared exogenous input. No learned-fit, actor-quality, inference-lookahead or depth-scaling claim.')
    out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))


if __name__=='__main__':main()
