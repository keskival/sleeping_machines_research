"""Outcome-independent sampling of bounded suffix utility positions."""
import torch


def sample_positions(length, count, generator):
    if length < 1 or count < 0:
        raise ValueError('Positive suffix length and nonnegative sample count required')
    # Zero selects the full-score control; do not advance randomness in that case.
    if count == 0 or count >= length:
        return torch.arange(length)
    if generator is None:
        raise ValueError('Position sampling needs a persistent independent generator')
    return torch.randperm(length, generator=generator)[:count].sort().values


@torch.no_grad()
def suffix_difference(readout, replay_features, targets, factual_nll, positions):
    """Estimate full per-lane suffix loss difference from a uniform position set."""
    if positions.ndim != 1 or positions.numel() == 0:
        raise ValueError('Nonempty scoring positions required')
    lanes, length = targets.shape
    if replay_features.shape[0] != lanes or replay_features.shape[1] <= int(positions.max()):
        raise ValueError('Replay must reach all scoring positions')
    alternative = readout.nll(replay_features[:, positions], targets[:, positions]).reshape(lanes, -1).mean(-1)
    factual = factual_nll.detach().reshape(lanes, length)[:, positions].mean(-1)
    return alternative.double() - factual.double()
