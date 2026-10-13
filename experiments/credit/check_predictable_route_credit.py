"""R1 learned residual integration gate: exhaustive tiny-route expectations."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sleeping_machines.predictable_route_credit import predictable_route_credit,admitted_prediction_loss
from sleeping_machines.paired_route_credit import paired_route_credit

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args();torch.set_num_threads(1)
 winner=torch.tensor([0]);delta=torch.tensor([[0.,2.,-.7]],dtype=torch.float64);q=torch.tensor([.3,.7],dtype=torch.float64);rows=[]
 for p in (.25,1.):
  for shared in (False,True):
   expected=torch.zeros(1,3,dtype=torch.float64);critic_expected=torch.zeros(1,3,dtype=torch.float64)
   for j,mass in ((None,1-p),(1,p*.3),(2,p*.7)):
    logits=torch.tensor([[.2,-.4,.8]],dtype=torch.float64,requires_grad=True);pi=logits.softmax(-1)
    pred=3*logits if shared else torch.tensor([[4.,.6,-1.2]],dtype=torch.float64,requires_grad=True)
    kw={} if j is None else dict(alternative=torch.tensor([j]),proposal_probability=q[j-1:j],loss_difference=delta[:,j])
    credit=predictable_route_credit(pi,pred,winner,admitted=j is not None,admission_probability=p,**kw)
    expected+=mass*torch.autograd.grad(credit,logits)[0]
    if not shared and j is not None:
     loss=admitted_prediction_loss(pred,winner,kw['alternative'],kw['proposal_probability'],kw['loss_difference'],p)
     critic_expected+=mass*torch.autograd.grad(loss,pred)[0]
   logits=torch.tensor([[.2,-.4,.8]],dtype=torch.float64,requires_grad=True);pi=logits.softmax(-1)
   exact=torch.autograd.grad(((pi-pi.detach())*delta).sum(),logits)[0]
   error=float((expected-exact).abs().max());assert error<1e-14
   if not shared:
    pred=torch.tensor([[4.,.6,-1.2]],dtype=torch.float64,requires_grad=True)
    exact_predictor=torch.autograd.grad(((pred[:,1:]-delta[:,1:])**2).sum(),pred)[0]
    prediction_error=float((critic_expected-exact_predictor).abs().max());assert prediction_error<1e-14
   else:prediction_error=None
   rows.append(dict(probability=p,shared_parameters=shared,conditional_credit_gradient_error=error,conditional_prediction_gradient_error=prediction_error))
 logits=torch.tensor([[.2,-.4,.8]],dtype=torch.float64,requires_grad=True);pi=logits.softmax(-1);zero=torch.zeros_like(pi);alt=torch.tensor([1])
 new=predictable_route_credit(pi,zero,winner,admitted=True,admission_probability=1,alternative=alt,proposal_probability=q[:1],loss_difference=delta[:,1])
 old=paired_route_credit(pi,alt,q[:1],delta[:,1]);newgrad=torch.autograd.grad(new,logits,retain_graph=True)[0];oldgrad=torch.autograd.grad(old,logits)[0];assert torch.equal(newgrad,oldgrad)
 names=[str(Path(__file__).relative_to(ROOT)),'sleeping_machines/predictable_route_credit.py','sleeping_machines/paired_route_credit.py','sleeping_machines/__init__.py']
 sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
 d=dict(status='completed',battle='R1',gate='PASS',rows=rows,zero_prediction_reproduces_existing_credit=True,source_sha256=sources,scope='Exhaustive three-route float64 numerical contract, probabilities0.25/1, arbitrary prediction and shared/separate leaves. No model fit, quality/work result or optimal admission claim. Admitted weighted prediction loss reproduces full-alternative supervised risk gradient; actor predictions detached.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
 print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
