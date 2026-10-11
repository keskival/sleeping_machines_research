"""Native feature tilt equals independent supervised gradient differences."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from token_issued_credit_geometry import route_mean_shifts
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();checks=[]
    for vocabulary in [23,50257]:
        torch.manual_seed(99400+vocabulary);d=8 if vocabulary==23 else 64
        embedding=torch.randn(vocabulary,d)*.2;logits=torch.randn(vocabulary)
        receivers=[0,2,vocabulary-1];beta=torch.tensor([0.,.4,-.3])
        shift,dp,mu=route_mean_shifts(logits,embedding,receivers,beta)
        errors=[]
        for target in [1,3]:
            z=torch.zeros(d,requires_grad=True)
            original=torch.nn.functional.cross_entropy((logits+embedding@z)[None],torch.tensor([target]))
            gradient=torch.autograd.grad(original,z)[0]
            for i,r in enumerate(receivers):
                changed=logits.clone();changed[r]+=beta[i]
                loss=torch.nn.functional.cross_entropy((changed+embedding@z)[None],torch.tensor([target]))
                newgrad=torch.autograd.grad(loss,z)[0]
                errors.append(float((newgrad-gradient-shift[i]).abs().max()))
        assert max(errors)<1e-12 and torch.equal(shift[0],torch.zeros(d))
        checks.append(dict(vocabulary=vocabulary,gradient_error=max(errors),label_independent=True))
    paths=['experiments/credit/token_issued_credit_geometry.py','experiments/credit/check_token_credit_geometry.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,checks=checks,whole_wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Independent full-vocabulary supervised gradients before/after single-address route tilt match prefix-only feature mean shift at23/50257 words and two labels, including no-op. This supplies a causal inductive feature, not future-label prediction or learned optimizer evidence.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
