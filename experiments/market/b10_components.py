"""Inference-only decomposition of the existing B10 recording-cell protocol."""
import torch
from b10_tpp import CELL, K, Poisson, Hawkes, Race, log_interval


def components(model, t, marks):
    gap = t[:, 1:] - t[:, :-1]
    if isinstance(model, Poisson):
        total = model.log_lam.exp().sum()
        time = log_interval(-total * gap, -total * (gap + CELL))
        mark = (model.log_lam - torch.logsumexp(model.log_lam, 0))[marks[:, 1:]]
        return time, mark
    if isinstance(model, Race):
        h, slots = model.m.encode(t, marks, torch.ones_like(marks, dtype=torch.bool))
        parameters = model.m.clocks(h[:, :-1], slots[:, :-1])
        gap = gap.clamp_min(0)
        _, survival0 = model.m.clock_terms(gap, *parameters[:4])
        _, survival1 = model.m.clock_terms(gap + CELL, *parameters[:4])
        hazard, _ = model.m.clock_terms(gap + CELL / 2, *parameters[:4])
        label = parameters[4].gather(-1, marks[:, 1:, None, None].expand(-1, -1, model.m.M, 1)).squeeze(-1)
        return (log_interval(survival0.sum(-1), survival1.sum(-1)),
                torch.logsumexp(hazard + label, -1) - torch.logsumexp(hazard, -1))
    if not isinstance(model, Hawkes):
        raise TypeError('Only the three registered B10 models are supported')
    base, amplitude, beta = model.log_mu.exp(), model.log_alpha.exp(), model.log_beta.exp()
    onehot = torch.nn.functional.one_hot(marks, K).to(t.dtype)
    state = onehot[:, 0].unsqueeze(1).expand(len(t), len(beta), K).clone()
    times, labels = [], []
    for index in range(1, t.shape[1]):
        if index > 1:
            elapsed = (t[:, index - 1] - t[:, index - 2]).view(-1, 1, 1)
            state = state * torch.exp(-beta.view(1, -1, 1) * elapsed) + onehot[:, index - 1].unsqueeze(1)
        excitation = torch.einsum('rkm,brm->brk', amplitude, state)
        g = gap[:, index - 1:index]

        def compensator(x):
            return base.sum() * x + (excitation.sum(-1) * (1 - torch.exp(-beta * x))).sum(-1, keepdim=True)

        intensity = base + (excitation * beta.view(1, -1, 1) *
                            torch.exp(-beta.view(1, -1, 1) * (g + CELL / 2).unsqueeze(-1))).sum(1)
        times.append(log_interval(-compensator(g), -compensator(g + CELL)).squeeze(1))
        labels.append((torch.log(intensity.gather(1, marks[:, index:index + 1])) -
                       torch.log(intensity.sum(1, keepdim=True))).squeeze(1))
    return torch.stack(times, 1), torch.stack(labels, 1)
