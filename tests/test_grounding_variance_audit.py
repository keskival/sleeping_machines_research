import torch
from experiments.hierarchy.grounding_variance_audit import constant_reference


def test_reference_equal_scene_weight_and_no_validation_fitting():
    train=[torch.tensor([[0.],[0.],[0.]],dtype=torch.float64),torch.tensor([[2.]],dtype=torch.float64)]
    dev=[torch.tensor([[4.]],dtype=torch.float64)]
    record,center=constant_reference(train,dev)
    assert center.item()==1 and record['train_loss']==1 and record['dev_loss']==9
    altered,_=constant_reference(train,[torch.tensor([[100.]],dtype=torch.float64)])
    assert altered['train_loss']==record['train_loss']
