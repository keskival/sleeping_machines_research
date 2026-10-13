"""Native temporal delayed Jacobian factors through four real state coordinates."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/tpp'))
from race_tpp_v19 import TemporalMemoryLayer


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(413);started=time.monotonic()
    layer=TemporalMemoryLayer(8,2,0.).double().eval();x=torch.randn(2,8,dtype=torch.float64);dt=torch.tensor([[0.,.37]],dtype=torch.float64)
    def write(u):
        w=layer.write(u);g=torch.sigmoid(layer.gate(u))
        return torch.cat((w[:2]*g,w[2:]*g))
    def read(s,u):
        h=layer.norm1(u+layer.read(s));return layer.norm2(h+layer.mlp(h))
    rate=layer.log_rate.exp();angle=layer.freq*dt[0,1];decay=torch.exp(-rate*dt[0,1])
    cr=decay*torch.cos(angle);ci=decay*torch.sin(angle)
    T=torch.cat((torch.cat((torch.diag(cr),-torch.diag(ci)),1),torch.cat((torch.diag(ci),torch.diag(cr)),1)),0)
    s1=write(x[0]);s2=T@s1+write(x[1])
    local=torch.stack((read(s1,x[0]),read(s2,x[1])))
    actual=layer(x[None],dt)[0];forward_error=float((local-actual).abs().max());assert forward_error<1e-12
    B=torch.func.jacrev(write)(x[0]);C=torch.func.jacrev(lambda s:read(s,x[1]))(s2)
    J=torch.func.jacrev(lambda u:layer(u.reshape(1,2,8),dt).flatten())(x.flatten())
    delayed=J[8:,:8];factor=C@T@B
    factor_error=float((factor-delayed).abs().max());assert factor_error<1e-12
    assert float(J[:8,8:].abs().max())==0
    c=torch.randn(8,dtype=torch.float64);v=torch.randn(8,dtype=torch.float64)
    credit=B.T@T.T@C.T@c;credit_error=float((credit-delayed.T@c).abs().max());assert credit_error<1e-12
    bilinear_error=float((v@credit-c@(factor@v)).abs());assert bilinear_error<1e-12
    singular=torch.linalg.svdvals(delayed);rank=int((singular>1e-10).sum());assert rank<=4
    sources=('experiments/credit/check_state_interface_credit.py','experiments/tpp/race_tpp_v19.py')
    result=dict(status='completed',battle='B1/R1',input_width=8,state_interface_width=4,delayed_rank=rank,
        singular_values=singular.tolist(),forward_error=forward_error,factor_error=factor_error,
        credit_error=credit_error,forward_adjoint_bilinear_error=bilinear_error,
        factor_tensor_scalars=B.numel()+T.numel()+C.numel(),dense_delayed_matrix_scalars=delayed.numel(),
        factor_matvec_multiply_add_flops=2*(B.numel()+T.numel()+C.numel()),dense_matvec_multiply_add_flops=2*delayed.numel(),
        wall_s=time.monotonic()-started,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
        scope='Fixed-parameter native two-event temporal layer; exact local endpoint derivatives plus decay/rotation factor delayed credit through four real state coordinates. Full reverse/JVP audit paid, not a task fit. Dense transport64floats versus unfused factors80floats: this short witness does not itself save storage/work. Learning/reuse across longer histories and whole learner-state cuts requires integrated quality/full accounting. No hard-route alternative credit or future-cotangent estimator trained here.')
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
