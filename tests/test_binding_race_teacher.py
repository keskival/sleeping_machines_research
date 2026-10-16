import torch
import pytest
from sleeping_machines.binding_race_teacher import BindingRaceTeacher


def example():
    torch.manual_seed(51)
    return BindingRaceTeacher(8, 4), torch.randn(8, requires_grad=True), torch.randn(3, 8, requires_grad=True), torch.tensor([[0.,0.],[1.,0.],[0.,1.]])


def test_exact_clock_normalization_and_local_credit():
    t, parent, children, poses = example()
    logits = t.logits(parent, children, poses)
    loss, work = t.loss(parent, children, poses)
    assert work['candidates'] == 4
    assert torch.allclose(logits.exp()/logits.exp().sum(), logits.softmax(0))
    loss.backward()
    assert parent.grad is not None and parent.grad.norm() > 0
    assert children.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in t.parameters())


def test_reenumeration_and_translation_do_not_change_target_race():
    t, parent, children, poses = example()
    original = t.logits(parent, children, poses)
    order = [2,0,1]
    changed = t.logits(parent, children[order], poses[order]+3)
    assert torch.allclose(original[0], changed[0], atol=1e-6)
    assert torch.allclose(original.sort().values, changed.sort().values, atol=1e-6)


def test_content_pose_reassignment_changes_positive_key():
    t, parent, children, poses = example()
    original = t.logits(parent, children, poses)[0]
    swapped = t.logits(parent, children[[1,0,2]], poses)[0]
    assert not torch.allclose(original, swapped)


def test_indistinguishable_swaps_are_not_false_negatives():
    t, parent, children, poses = example()
    loss, work = t.loss(parent, children[:1].expand(3,-1), poses)
    assert not work['informative'] and work['candidates'] == 1 and loss == 0


def test_incomplete_input_is_rejected():
    t, parent, children, poses = example()
    with pytest.raises(ValueError): t.loss(parent, children[:1], poses[:1])
