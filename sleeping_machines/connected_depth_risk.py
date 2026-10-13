"""Cross-audit projected risk for connected, version-bound depth credit.

Inputs are already transported residual contributions, not local cotangents.
The caller must select the two sites independently with positive predictable
probabilities, on the same issued path/query, and retain the issued graph.
Transport, anchors, counterfactual route terms and probe acquisition are not
implemented or made free by this scalar objective.
"""
import torch


def cross_audit_risk(first, second, first_probability, second_probability):
    if first.shape != second.shape or first.ndim < 1:
        raise ValueError('Matched projected residual contributions required')
    p = torch.as_tensor(first_probability, dtype=first.dtype, device=first.device).detach()
    q = torch.as_tensor(second_probability, dtype=second.dtype, device=second.device).detach()
    if p.numel() != 1 or q.numel() != 1:
        raise ValueError('One recorded site probability per audit required')
    if not bool(torch.isfinite(p).all() and torch.isfinite(q).all() and (p > 0).all() and (q > 0).all() and (p <= 1).all() and (q <= 1).all()):
        raise ValueError('Finite positive issued probabilities required')
    return ((first/p)*(second/q)).sum(-1).mean()


def check_cross_audit_contract():
    """Enumerate all site pairs: exact aggregate risk and joint gradients."""
    for depth in (8, 16):
        values = torch.arange(depth*3, dtype=torch.float64).reshape(depth, 3)/19-1
        values.requires_grad_()
        p = torch.arange(1, depth+1, dtype=torch.float64);p = p/p.sum()
        q = p.flip(0)
        expected = sum(p[i]*q[j]*cross_audit_risk(values[i], values[j], p[i], q[j])
                       for i in range(depth) for j in range(depth))
        exact = values.sum(0).square().sum()
        assert torch.allclose(expected, exact, atol=1e-12, rtol=1e-12)
        observed_gradient = torch.autograd.grad(expected, values, retain_graph=True)[0]
        exact_gradient = torch.autograd.grad(exact, values)[0]
        assert torch.allclose(observed_gradient, exact_gradient, atol=1e-12, rtol=1e-12)
    opposite = torch.tensor([[2., -3.], [-2., 3.]], dtype=torch.float64)
    p = .5
    expected = sum(p*p*cross_audit_risk(opposite[i], opposite[j], p, p)
                   for i in range(2) for j in range(2))
    assert float(expected) == 0 and float(opposite.square().sum()) > 0
    # Individual unbiased samples may be negative; clipping the sample loss
    # would destroy the aggregate-risk identity.
    assert float(cross_audit_risk(opposite[0], opposite[1], p, p)) < 0
