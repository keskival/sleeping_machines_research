"""R1/B1 forward-probe risk and sparse correction on actual native depth cuts."""
import argparse,hashlib,itertools,json,time
import torch
from check_deep_temporal_packet import ROOT,native
from native_deep_temporal_packet import NativeDeepTemporalPacket

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);rows=[]
    for depth in [2,4,8]:
        torch.manual_seed(99580+depth);layers=torch.nn.ModuleList([native.TemporalMemoryLayer(4,3,0.).eval() for _ in range(depth)])
        x=torch.randn(1,2,4);elapsed=torch.tensor([[0.,.7]]);packet=NativeDeepTemporalPacket.capture(layers,x,elapsed,context='credit-probe',version=0)
        c=torch.randn(1,1,2,4);bx,bt=packet.transpose_with_time(c,context='credit-probe',version=0);b=torch.cat([bx.flatten(),bt.flatten()]);n=len(b)
        directions=torch.tensor(list(itertools.product([-1.,1.],repeat=n)))
        responses=packet.apply_with_time(directions[:,:8].reshape(-1,1,2,4),directions[:,8:].reshape(-1,1,2),context='credit-probe',version=0)
        y=(responses*c).flatten(1).sum(1);reciprocity=float((y-directions@b).abs().max())
        h=(b+torch.randn_like(b)*.1).requires_grad_();risk=(directions@h-y).square().mean();expected=(h-b).square().sum()
        grad=torch.autograd.grad(risk,h)[0];gtrue=2*(h.detach()-b)
        corrected=h.detach()[None]+directions*(y-directions@h.detach())[:,None]
        mean_error=float((corrected.mean(0)-b).abs().max());variance=(corrected-b).square().sum(1).mean();theory=(n-1)*(h.detach()-b).square().sum()
        errors=dict(reciprocity=reciprocity,risk=float((risk-expected).abs().detach()),gradient=float((grad-gtrue).abs().max()),corrected_mean=mean_error,variance=float((variance-theory).abs()))
        assert max(errors.values())<1e-10,errors
        rows.append(dict(depth=depth,cut_dimension=n,enumerated_directions=len(directions),errors=errors,correction_variance=float(variance),residual_squared_norm=float(expected.detach()),cached_floats=packet.cached_floats()))
    paths=['experiments/credit/check_forward_probe_credit_risk.py','experiments/credit/native_deep_temporal_packet.py','experiments/credit/native_temporal_teaching_packet.py','experiments/credit/native_cached_linearization.py','experiments/tpp/race_tpp_v19.py']
    result=dict(status='completed',battle='R1/B1',tag=a.tag,gate=True,rows=rows,wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Real native temporal memory layers/depth2/4/8, synthetic2eventcontent/time cut, exact enumeration1024Rademacher directions. Validates forward-supervised cotangent risk, parameter-of-output gradient and mean/variance correction identities. No learned-critic fit, task quality, cost advantage or genericdepth stability claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
