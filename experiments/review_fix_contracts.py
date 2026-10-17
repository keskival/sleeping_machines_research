#!/usr/bin/env python3
"""Contracts for the 17 Oct flagship review fixes (r1_cerl.param_groups, unit_init, balance; used by r1_cerl/r1_rmr/r1_cerl_gpu).

1. Defaults are the original runs: groups=False is one group over all parameters; unit_init=0 and balance=0 build the same
   parameters from the same seed and give identical NLL and auxiliary loss.
2. groups=True partitions the parameters exactly once; decayed tensors are all >= 2-D and none is a dormant-capacity table.
3. The load-balance term (flat bank; product keys: per half-bank, 2 sqrt(U) scores) is the Switch form balance*U*sum(f*P): it is >= balance with equality when routing is uniform, and it
   only adds to the auxiliary objective (the NLL, i.e. the predictive race, is unchanged).
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
import r1_cerl as C  # noqa: E402
import r1_rmr as RM  # noqa: E402


def build(seed, **kw):
    torch.manual_seed(seed); counts = np.random.default_rng(0).integers(1, 1000, 512).astype(float); cls, words = C.word_classes(counts, 16)
    return RM.RMR(512, cls, words, dict(d=32, modes=8, layers=2, dk=16, units=64, topk=4, explore=2, **kw), levels=3, dk=16, top_m=2, chunk=16, rezero=True, old_decay=0.01)


def main():
    out = {}; torch.set_num_threads(1); x = torch.randint(0, 512, (2, 49)); xi, yi = x[:, :-1], x[:, 1:]
    a, b = build(1), build(1, unit_init=0.0, balance=0.0)
    same = all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    torch.manual_seed(5); na = a.lane_nll(xi, yi); aa = a.aux; torch.manual_seed(5); nb = b.lane_nll(xi, yi); ab = b.aux
    out['defaults_identical'] = dict(params=same, nll_diff=float((na - nb).abs().max()), aux_diff=float((aa - ab).abs().max()))
    g0 = C.param_groups(a, 0.01, False); out['default_one_group'] = len(g0) == 1 and len(g0[0]['params']) == len(list(a.parameters()))
    g1 = C.param_groups(a, 0.01, True); ids = [id(p) for g in g1 for p in g['params']]
    names = {id(p): n for n, p in a.named_parameters()}; dec = [names[id(p)] for p in g1[0]['params']]; nod = [names[id(p)] for p in g1[1]['params']]
    out['groups_partition'] = len(ids) == len(set(ids)) == len(list(a.parameters()))
    out['decayed_all_matrices_not_dormant'] = all(dict(a.named_parameters())[n].dim() >= 2 and n.split('.')[-1] not in ('unit_key', 'unit_out', 'unit_val', 'key1', 'key2') for n in dec)
    out['no_decay'] = nod
    c = build(1, unit_init=0.02, balance=0.1); c.train(); torch.manual_seed(5); nc = c.lane_nll(xi, yi); ac = c.aux
    d_ = build(1, unit_init=0.02, balance=0.0); d_.train(); torch.manual_seed(5); nd = d_.lane_nll(xi, yi); ad = d_.aux
    bal = float((ac - ad).mean()); out['balance'] = dict(nll_unchanged=float((nc - nd).abs().max()), term=bal, at_least_balance=bal >= 0.1 - 1e-6)
    out['unit_out_symmetry_broken'] = float(c.base.unit_out.std()) > 0
    torch.manual_seed(2); counts = np.random.default_rng(0).integers(1, 1000, 512).astype(float); cls, words = C.word_classes(counts, 16)
    pk = [C.CERL(512, cls, words, 32, 8, 1, 16, 64, 4, 0, product_keys=True, balance=bb) for bb in (0.1, 0.0)]; pk[1].load_state_dict(pk[0].state_dict())
    for m_ in pk: m_.train()
    n1, n0 = pk[0].lane_nll(xi, yi), pk[1].lane_nll(xi, yi); a1, a0 = pk[0].aux, pk[1].aux
    tpk = float((a1 if a0 is None else a1 - a0).mean()); out['pk_balance'] = dict(nll_unchanged=float((n1 - n0).abs().max()), term=tpk, at_least_balance=tpk >= 0.1 - 1e-6)
    a1.mean().backward(); out['pk_balance']['credit_to_half_keys'] = float(pk[0].key1.grad.abs().max()) > 0
    ok_pk = out['pk_balance']['nll_unchanged'] < 1e-6 and out['pk_balance']['at_least_balance'] and out['pk_balance']['credit_to_half_keys']
    ok = ok_pk and same and out['defaults_identical']['nll_diff'] == 0 and out['defaults_identical']['aux_diff'] == 0 and out['default_one_group'] and out['groups_partition'] \
        and out['decayed_all_matrices_not_dormant'] and out['balance']['nll_unchanged'] < 1e-6 and out['balance']['at_least_balance'] and out['unit_out_symmetry_broken']
    out['status'] = 'passed' if ok else 'failed'
    out['source_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/r1_cerl.py', 'experiments/r1_rmr.py', 'experiments/review_fix_contracts.py')}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
