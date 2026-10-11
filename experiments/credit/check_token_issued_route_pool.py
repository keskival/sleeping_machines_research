"""Full-prefix versus causal carried history pool and actor-probability parity."""
import argparse,hashlib,json,time
import torch
from check_token_forward_credit import ROOT,KeyedTokenLM
from token_issued_route_features import issue,RouteController
from token_issued_route_pool import issue_with_history

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();checks=[]
    for depth in [2,4,8]:
        torch.manual_seed(9750+depth);m=KeyedTokenLM(8,3,depth,2,4,1,0.)
        m.embed=torch.nn.Embedding(23,8);m.bias=torch.nn.Parameter(torch.randn(23)*.02)
        m.requires_grad_(False);m.eval();history=torch.tensor([2,4,6,8,10,12,14,16,2,4])
        z=torch.randn(8)*.1;full=issue(m,history,fast_state=z)
        before=issue(m,history[:-1],fast_state=z)
        carried=issue_with_history(m,history[-1:],history,before['carrier'],z)
        assert full['receivers']==carried['receivers'] and torch.equal(full['actions'],carried['actions'])
        errors={n:float((full[n]-carried[n]).abs().max()) for n in ['context','descriptors']}
        actor=RouteController(len(full['context']),full['descriptors'].shape[1])
        errors['probabilities']=float((actor(full['context'],full['descriptors']).softmax(0)-actor(carried['context'],carried['descriptors']).softmax(0)).abs().max().detach())
        assert max(errors.values())<1e-12
        checks.append(dict(depth=depth,routes=len(full['receivers']),errors=errors))
    paths=['experiments/credit/check_token_issued_route_pool.py','experiments/credit/token_issued_route_pool.py','experiments/credit/token_issued_route_features.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,checks=checks,whole_wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Same observed history/candidate directions/transported descriptors/actor probabilities for full-prefix and carried-event issuance at depths2/4/8. Prefix-only API, no future inputs. GPT-2 EOS pool protocol; no task fitting/quality claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
