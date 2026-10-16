import torch
from torch.nn import functional as F
from sleeping_machines.composition_task_credit import actual_joint_loss_credit
from sleeping_machines.coupled_native_query_credit import ordered_pair_probabilities


def test_exact_actual_loss_estimator_matches_finite_sum_for_every_factual_pair():
 s=torch.tensor([[.3,-.4,.1],[-.2,.7,.2]],dtype=torch.float64,requires_grad=True)
 pi=ordered_pair_probabilities(s).flatten();loss=F.softplus(-torch.arange(9,dtype=torch.float64).sin()*3);valid=[(i,j) for i in range(3) for j in range(3) if i!=j]
 expected=torch.autograd.grad((pi*loss).sum(),s,retain_graph=True)[0]
 for w in valid:
  total=0.
  for a in valid:
   if a==w:continue
   j=a[0]*3+a[1];k=w[0]*3+w[1];q=(pi[j]/(1-pi[k])).detach()
   surrogate=actual_joint_loss_credit(s,w,a,loss[k],loss[j]);assert float(surrogate.detach())==0
   total=total+q*surrogate
  got=torch.autograd.grad(total,s,retain_graph=True)[0];torch.testing.assert_close(got,expected,atol=2e-14,rtol=2e-14)


def test_joint_nonlinear_suffix_has_real_credit_when_tangent_is_zero():
 # Relational suffix: neither local component alone explains their conjunction.
 z=torch.tensor([0.,0.],dtype=torch.float64,requires_grad=True);other=torch.ones_like(z)
 factual=F.softplus(-z.prod());alternative=F.softplus(-other.prod())
 tangent=(torch.autograd.grad(factual,z)[0]*(other-z)).sum()
 assert float(tangent)==0 and float(alternative-factual)<-.3
 s=torch.tensor([[.2,-.1],[.3,.4]],dtype=torch.float64,requires_grad=True)
 exact=actual_joint_loss_credit(s,(0,1),(1,0),factual,alternative)
 assert torch.autograd.grad(exact,s)[0].norm()>.01


def test_first_order_loss_teacher_can_point_in_the_wrong_direction():
 z=torch.tensor(0.,dtype=torch.float64,requires_grad=True);other=z.new_tensor(1.)
 loss=lambda x:F.softplus((x-.25).square())
 tangent=torch.autograd.grad(loss(z),z)[0]*(other-z)
 actual=loss(other)-loss(z)
 assert float(tangent)<0 and float(actual)>0
