"""Static credit from the issued fast-state snapshot, with causal EOS reset.
Unlike the legacy zero-fast-state projection, this control retains the learner
already acquired before issuance and freezes only subsequent adaptation.
"""
import torch
from token_forward_credit import bonuses, projected_costs


def static_carried_scores(packet, receivers, actions, initial_z):
    if initial_z.shape != (packet.embedding.shape[1],):
        raise ValueError('Fast state does not match the issued embedding')
    beta = bonuses(packet, actions)
    with torch.no_grad():
        offset = packet.embedding @ initial_z.detach()
        rows = []
        for j, target in enumerate(packet.targets.tolist()):
            if bool(packet.reset[j]):
                offset = torch.zeros_like(offset)
            rows.append(projected_costs(packet.log_probabilities[j] + offset,
                                        receivers, beta[:, j], target))
        return torch.stack(rows, 1)
