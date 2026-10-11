"""R1 dense decoder accounting parity for mv and equivalent two-dimensional mm."""
import argparse,hashlib,json
from pathlib import Path
import torch
from torch.utils.flop_counter import FlopCounterMode
from token_flop_accounting import TokenFlopCounter
ROOT=Path(__file__).resolve().parents[2]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    matrix=torch.arange(77,dtype=torch.float64).reshape(11,7)/77;z=torch.ones(7,requires_grad=True)
    with TokenFlopCounter() as counter:(matrix@z).sum().backward()
    actual=counter.get_total_flops();expected=4*11*7;assert actual==expected,(actual,expected)
    zz=torch.ones(1,7,requires_grad=True)
    with FlopCounterMode(display=False) as mm:(zz@matrix.T).sum().backward()
    assert mm.get_total_flops()==expected and torch.equal(z.grad,zz.grad[0])
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,mv_forward_backward_flops=actual,mm_forward_backward_flops=mm.get_total_flops(),expected=expected,gradient_error=float((z.grad-zz.grad[0]).abs().max()),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/credit/token_flop_accounting.py','experiments/credit/check_token_flop_accounting.py']},scope='Exact dense matrix-vector2VD operation convention and backward accounting; other unsupported scalar/special/optimizer operations stay omitted from supported FLOPs and included in wall. No training or quality claim.')
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
