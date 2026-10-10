"""B1/R1 reciprocal teaching: exhaustive identities and native temporal JVPs."""
import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/tpp'))
from race_tpp_v19 import TemporalMemoryLayer


def enumeration(J, rng):
    m, n = J.shape; B = rng.normal(size=(n, m)); D = B-J.T
    beta = 1/(n*m); losses = []; updates = []; errors = []
    for i in range(n):
        for j in range(m):
            v = np.eye(n)[i]*np.sqrt(n); lam = np.eye(m)[j]*np.sqrt(m)
            delta = v@B@lam-lam@J@v
            update = delta*np.outer(v, lam)
            losses.append(delta**2/2); updates.append(update)
            errors.append(np.sum((D-beta*update)**2))
    result = dict(shape=[n,m], beta=beta,
                  loss_identity_error=abs(np.mean(losses)-np.sum(D**2)/2),
                  mean_update_error=float(np.max(np.abs(np.mean(updates,axis=0)-D))),
                  contraction_identity_error=abs(np.mean(errors)-(1-2*beta+beta**2*n*m)*np.sum(D**2)))
    # One complete coordinate sweep is a bounded learning-rule contract, not
    # evidence for sparse discovery or efficient task fitting.
    initial = np.sum(D**2)
    for i in range(n):
        for j in range(m):
            v = np.eye(n)[i]*np.sqrt(n); lam = np.eye(m)[j]*np.sqrt(m)
            delta = v@B@lam-lam@J@v
            B -= beta*delta*np.outer(v,lam)
    result.update(initial_squared_error=float(initial), final_squared_error=float(np.sum((B-J.T)**2)))
    assert max(result[k] for k in ('loss_identity_error','mean_update_error','contraction_identity_error')) < 1e-9
    assert result['final_squared_error'] < 1e-20
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--tag',required=True); a=p.parse_args()
    start=time.monotonic(); torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    torch.manual_seed(211); rng=np.random.default_rng(211)
    synthetic=enumeration(rng.normal(size=(2,3)),rng)
    layer=TemporalMemoryLayer(8,4,0.0).double().eval()
    x=torch.randn(16); dt=torch.tensor([[0.0,0.37]])
    def forward(z): return layer(z.reshape(1,2,8),dt).flatten()
    basis=torch.eye(16); columns=[]; tick=time.monotonic()
    for v in basis:
        _, jv=torch.func.jvp(forward,(x,),(v,)); columns.append(jv.detach())
    J=torch.stack(columns,1); forward_probe_s=time.monotonic()-tick
    # Independent reverse audit verifies the forward probe labels.
    xx=x.detach().requires_grad_(); yy=forward(xx); rows=[]; tick=time.monotonic()
    for j in range(16): rows.append(torch.autograd.grad(yy[j],xx,retain_graph=j<15)[0])
    reverse=torch.stack(rows); reverse_audit_s=time.monotonic()-tick
    parity=float((J-reverse).abs().max()); assert parity<1e-10
    native=enumeration(J.numpy(),rng)
    W=rng.normal(size=(4,3)); B=rng.normal(size=(3,4)); D=B-W.T; alpha=.07
    for _ in range(20):
        update=rng.normal(size=W.shape)
        W=(1-alpha)*W+update; B=(1-alpha)*B+update.T
    mirror_error=float(np.max(np.abs((B-W.T)-(1-alpha)**20*D))); assert mirror_error<1e-12
    sources=['experiments/credit/reciprocal_probe_contract.py','experiments/tpp/race_tpp_v19.py']
    result=dict(status='completed',tag=a.tag,battle='B1/R1',stage='reciprocal_probe_contract',
                synthetic=synthetic,native_temporal=native,native_jvp_vjp_error=parity,
                mirrored_tracking_error=mirror_error,forward_probe_s=forward_probe_s,reverse_audit_s=reverse_audit_s,
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                hardware=dict(host=platform.node(),threads=1,dtype='float64'),
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='Exhaustive learning-rule identities and frozen native two-event temporal layer; all 16 tangent probes and 16 reverse audits charged. '
                      'No task fit, changing nonlinear actor, exploration utility, full model, TEST, efficiency or global convergence claim.',
                decision='Passing permits design of bounded changing-weight reciprocal integration; it does not admit scaling.')
    out=ROOT/'experiments/results/credit'/f'{a.tag}.json'
    with out.open('x') as f: f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
