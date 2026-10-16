import torch
from experiments.hierarchy.composition_credit_audit import pair_summary


def test_opposite_label_pair_credit_is_odd_vjp_at_zero_logits():
 a=torch.tensor([1.,2.,3.],dtype=torch.float64);b=torch.tensor([1.,4.,-1.],dtype=torch.float64)
 r=pair_summary(a,b,-.5,.5)
 assert r['decomposition_abs_error']<1e-15
 assert abs(r['credit_norm']-float((b-a).norm()/4))<1e-15
 r=pair_summary(a,a,-.5,.5);assert r['credit_norm']==0 and r['cancellation_ratio']==0


def test_additive_separate_child_features_have_zero_balanced_task_covariance():
 from experiments.hierarchy.part_combination_task import corpus
 rows=[r for r in corpus() if r.split=='train' and r.nuisance==0]
 torch.manual_seed(7187);left=torch.randn(8,2,19,dtype=torch.float64);right=torch.randn(8,2,19,dtype=torch.float64)
 x=torch.stack([left[r.left,r.root_swap]+right[r.right,r.root_swap] for r in rows]);y=torch.tensor([r.label for r in rows],dtype=torch.float64)
 # Logistic risk is convex;zero gradient at the constant predictor is a global
 # optimum for any such fixed additive representation (with intercept).
 assert ((.5-y)[:,None]*x).mean(0).abs().max()<1e-15
 assert abs(float((.5-y).mean()))<1e-15
