"""Issued context parity between full prefix and carried event execution."""
import argparse,hashlib,json,time
import torch
from check_token_forward_credit import ROOT,KeyedTokenLM
from token_issued_route_features import issue,RouteController

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();checks=[]
    for depth in [2,4,8]:
        torch.manual_seed(9200+depth);m=KeyedTokenLM(8,3,depth,2,4,1,0.)
        m.embed=torch.nn.Embedding(23,8);m.bias=torch.nn.Parameter(torch.randn(23)*.02)
        m.requires_grad_(False);m.eval();tokens=torch.tensor([2,4,6,8,2,4,6,8])
        z=torch.randn(8)*.1;full=issue(m,tokens,fast_state=z)
        before=issue(m,tokens[:-1],fast_state=z)
        carried=issue(m,tokens[-1:],initial_state=before['carrier'],fast_state=z)
        # Current-event features agree; candidate discovery needs full observed
        # document history, so its bounded pool intentionally differs here.
        error=float((full['context']-carried['context']).abs().max());assert error<1e-12
        actor=RouteController(len(full['context']),full['descriptors'].shape[1])
        logits=actor(full['context'],full['descriptors']);assert torch.isfinite(logits).all()
        logits.square().sum().backward();assert all(p.grad is not None for p in actor.parameters())
        assert not full['context'].requires_grad and not full['descriptors'].requires_grad
        checks.append(dict(depth=depth,context_error=error,routes=len(logits),actor_parameters=sum(p.numel() for p in actor.parameters())))
    paths=['experiments/credit/token_issued_route_features.py','experiments/credit/check_token_issued_route_features.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,checks=checks,whole_wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Prefix-only API; depth2/4/8 issued context parity with continuous native state, finite connected actor gradients, detached source features. Candidate discovery must retain observed document history in integration. No fitting or full actor suffix-invariance claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
