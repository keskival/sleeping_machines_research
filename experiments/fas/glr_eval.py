"""Evaluate a trained FAS race-readout checkpoint with the per-step slowdown GLR rule (THEORY §440.3) beside the
declared NLL rules, on validation only (battle B3 development; evaluation only).

Reports validation AUROC by merged prefix, overall and per fault type, for rules total, type, gap and glr_max.
Saves per-run scores to <tag>_scores.npz.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments')); sys.path.insert(0, str(ROOT / 'experiments/fas'))
from native import PREFIXES, V, load  # noqa: E402
from native_race_readout import scores  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402
from sleeping_machines.race_readout import BindingMemory, RaceReadout  # noqa: E402

OUT = ROOT / 'experiments/results/fas'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True); p.add_argument('--runs', type=int, default=1000)
    p.add_argument('--lanes', type=int, default=32); p.add_argument('--tag', required=True)
    a = p.parse_args()
    out = Path(OUT) / f'{a.tag}.json'   # aws_benchmark redirects OUT as a string
    if out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1)
    res = json.loads(Path(a.result).read_text()); args = res['args']
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=V, classes=V + 2, payload=args['payload'],
                                            depth=args['depth'], heads=args['heads'], pool=args['pool'])
    ctx = dict(full=True, additive='additive', none=False)[args.get('readout_context', 'full')]
    binding = None
    if args.get('binding_slots'):
        readout = RaceReadout(V, 1, args['binding_slots'], args['payload'], args['heads'] * args['payload'], hidden=args['hidden'],
                              type_durations=args.get('type_durations', False), classes=args.get('step_classes', 0), context=ctx)
        binding = BindingMemory(args['heads'] * args['payload'], args['binding_slots'], args['payload'],
                                tau_max=args.get('tau_max') or 1000., gated=args.get('binding_gated', False))
    else:
        readout = RaceReadout(V, args['heads'], args['pool'], args['payload'], args['heads'] * args['payload'], hidden=args['hidden'],
                              type_durations=args.get('type_durations', False), classes=args.get('step_classes', 0), context=ctx)
    w = Path(res['selected_weights']); ck = torch.load(w if w.is_absolute() else ROOT / w, weights_only=True)
    model.load_state_dict(ck['model']); readout.load_state_dict(ck['readout'])
    if binding is not None:
        binding.load_state_dict(ck['binding'])
    cell = args['cell_ms'] / 1000. if args.get('cell_ms') else None
    d = ROOT / 'experiments/data/fas' / args['data']
    vc, _ = load(d / 'val_clean.npz', args['max_events']); vf, vk = load(d / 'val_faulty.npz', args['max_events'])
    vc, vf, vk = vc[:a.runs], vf[:a.runs], vk[:a.runs]
    sc_c, nll, _ = scores(model, readout, vc, a.lanes, args['posterior'], False, binding, cell, glr=True)
    sc_f, _, _ = scores(model, readout, vf, a.lanes, args['posterior'], False, binding, cell, glr=True)
    result = dict(status='completed', battle='B3', source_result=a.result, runs=a.runs, val_clean_nll=nll,
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/fas/glr_eval.py', 'experiments/fas/native_race_readout.py',
                                  'sleeping_machines/race_readout.py')})
    from baselines import auroc
    table = {}
    for r in sc_c:
        table[r] = {str(n): dict(all=auroc(sc_c[r][:, j], sc_f[r][:, j]),
                                 wear_and_tear=auroc(sc_c[r][:, j], sc_f[r][vk == 1, j]),
                                 retry_delay=auroc(sc_c[r][:, j], sc_f[r][vk == 2, j])) for j, n in enumerate(PREFIXES)}
    result['val_auroc'] = table
    np.savez_compressed(out.with_name(f'{a.tag}_scores.npz'), prefixes=np.array(PREFIXES), fault_kind=np.asarray(vk),
                        **{f'val_clean_{r}': v for r, v in sc_c.items()}, **{f'val_faulty_{r}': v for r, v in sc_f.items()})
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({r: {n: round(v['all'], 3) for n, v in t.items()} for r, t in table.items()}))


if __name__ == '__main__':
    main()
