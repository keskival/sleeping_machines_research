"""Conditional algebra, complete objective derivatives and old-write fallback."""
import copy
import numpy as np
import torch
from experiments.r1_rmr import RMR, word_classes
from experiments.r1_rmr_compact import CompactRMR


def models(top_m=0, read_topk=0):
    torch.manual_seed(19)
    cls, words = word_classes(np.arange(41, dtype=float) + 1, 5, s_max=16)
    ref = RMR(41, cls, words, dict(d=16, modes=4, layers=1, dk=8,
              units=16, topk=3, explore=2), levels=3, dk=8,
              top_m=top_m, chunk=7, read_topk=read_topk)
    with torch.no_grad():
        for level in ref.levels:
            level.alpha_new.fill_(.4)
            level.alpha_old.fill_(.3)
    compact = copy.deepcopy(ref)
    compact.__class__ = CompactRMR
    return ref, compact


def test_compact_full_objective_gradients_and_sparse_reads():
    for read_topk in [0, 3]:
        ref, compact = models(read_topk=read_topk)
        x = torch.randint(0, 11, (2, 23)); y = torch.randint(0, 41, x.shape)
        losses = []
        for model in [ref, compact]:
            torch.manual_seed(83)
            loss = model.lane_nll(x, y)
            (loss.mean() + model.aux.mean()).backward()
            losses.append(loss.detach())
        torch.testing.assert_close(*losses, rtol=2e-6, atol=2e-6)
        for (name, p), (other, q) in zip(ref.named_parameters(), compact.named_parameters()):
            assert name == other
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=3e-6, msg=name)
        assert compact._mem[1] == []
        assert len(ref._mem[1]) == 2


def test_compact_evaluation_causality_chunks_and_old_write_fallback():
    for top_m in [0, 2]:
        ref, compact = models(top_m=top_m)
        ref.eval(); compact.eval()
        x = torch.randint(0, 11, (1, 29)); y = torch.randint(0, 41, x.shape)
        with torch.no_grad():
            full = compact.lane_nll(x, y)
            torch.testing.assert_close(full, ref.lane_nll(x, y), rtol=2e-6, atol=2e-6)
            chunks = torch.cat([compact.lane_nll(x, y, (s, min(s+5, 29)))
                                for s in range(0, 29, 5)], 1)
            torch.testing.assert_close(full, chunks)
            changed = x.clone(); changed[:, 19:] = (changed[:, 19:] + 1) % 41
            torch.testing.assert_close(full[:, :19], compact.lane_nll(changed, y)[:, :19])
        assert bool(compact._mem[1]) == bool(top_m)
