"""Replay an addressed list of native Adam worlds without a route/action product."""
import torch

from optimizer_aware_route_credit import BranchReadout, native


def replay_pairs(source, t, m, receivers, actions, lr):
    if len(receivers) != len(actions) or not len(receivers):
        raise ValueError('Each addressed world needs one route and action')
    routes = torch.tensor(receivers)
    model = native.actor(len(source.layers), 0)
    model.load_state_dict(source.state_dict())
    model.requires_grad_(False)
    model.mark_ctx = BranchReadout(model.mark_ctx, len(routes))
    opt = torch.optim.Adam(model.mark_ctx.parameters(), lr=lr)
    tt, mm = t.repeat(len(routes), 1), m.repeat(len(routes), 1)
    scores = []
    for j in range(4, m.shape[1]):
        opt.zero_grad()
        loss, _ = native.losses(model, tt[:, :j+1], mm[:, :j+1], 3, routes, actions)
        current = loss[:, -1]
        assert torch.isfinite(current).all()
        scores.append(current.detach())
        current.sum().backward()
        opt.step()
    return torch.stack(scores, 1), {k: v.detach().clone() for k, v in model.mark_ctx.state_dict().items()}
