"""Public benchmark campaign (experiments/SOTA_TARGETS.md): NeuroBench Mackey-Glass, NeuroBench primate reaching, SHD.

Every number is read from completed result files.  Partial official runs are labelled with their repeat count; development
results are labelled as development (selection uses validation or other-tau series; official test protocols are run once).
Published reference values are cited from the NeuroBench leaderboard and the fmi-basel neural-decoding-RSNN results.
"""
import glob
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'experiments/results'
MG_REFERENCES = [('LSTM (NeuroBench)', 13.37, 4.90e5, 6.03e4), ('ESN (NeuroBench)', 14.79, 2.81e5, 4.37e3)]
PRIMATE_LEADERBOARD = [('AEGRU', .71, 45500), ('GRU-t1', .707, 352904), ('bigSNN (bigRSNN)', .698, 4833360),
                       ('tinyRSNN', .66, 27144)]
PRIMATE_DEV_SESSION = {'bigRSNN': .772, 'tinyRSNN': .746}          # indy_20170131_02, fmi-basel results_summary_*.json


def load(read=None):
    aggregate = runpy.run_path(str(ROOT/'experiments/public_benchmarks/collect_neurobench_mg.py'))['collect']()
    official = [row for p in aggregate['parents'] for row in json.loads((ROOT/p['path']).read_text())['repeats']]
    mg = dict(n=aggregate['completed_repeats'], modes=aggregate['mean_smape_by_mode'],
              protocol_claim_eligible=aggregate['protocol_claim_eligible'], parents=aggregate['parents'],
              spread=(min(x['smape_by_mode']['mix8'] for x in official), max(x['smape_by_mode']['mix8'] for x in official))
              if official else None)
    work = json.loads((RES / 'neurobench_mg/curie_mg_inference_work_20261004T190000Z.json').read_text())['rows']
    mg['work'] = {(w['payload'], w['pool']): w for w in work}
    dev = []
    for f in sorted(glob.glob(str(RES / 'neurobench_mg/curie_mg*_t1[89]_*.json'))):
        r = json.loads(Path(f).read_text())
        if r.get('status') != 'completed': continue
        a = r['args']
        by_mode = r.get('mean_smape_by_mode') or {('argmax' if a.get('argmax') else 'sampled'): r['mean_smape']}
        member = r.get('member', 'race (sampled routing)')
        label = (f"tau {a['tau']}, p{a['payload']}/d{a['depth']}/pool{a['pool']}" + (', increments' if a.get('delta') else '')
                 + (', closed-loop' if a.get('closed_loop') else '') + (', det. training' if a.get('train_argmax') else '')
                 + (f", {a['steps']} steps" if a.get('steps', 1500) != 1500 else ''))
        dev.append(dict(label=label, member='expected reception' if 'expected' in member else 'race',
                        modes=by_mode, repeats=len(r['repeats'])))
    primate = []
    for f in sorted(glob.glob(str(RES / 'neurobench_primate/curie_pr*.json'))):
        r = json.loads(Path(f).read_text())
        if r.get('status') != 'completed': continue
        a = r['args']; s = r['sessions'][0]
        label = (f"p{a['payload']}/d{a['depth']}/pool{a['pool']}" + (' tied' if a.get('tie_pools') else '')
                 + (f", wd {a['weight_decay']:g}" if a.get('weight_decay') else '') + f", {a['steps']} steps"
                 + (', leaky readout' if a.get('smoothing') else '') + (f", {a['route_samples']} route samples"
                                                                       if a.get('route_samples', 1) > 1 else '')
                 + (', traces' if a.get('traces') else '') + (', expected reception' if a.get('reception') == 'expected' else ''))
        primate.append(dict(label=label, test=s['test_r2'], val=s['val_r2'], val_blocks=a.get('val_blocks', 1),
                            params=s['parameters']))
    return dict(mg=mg, mg_dev=dev, primate=primate)


def pages(data):
    mg, dev, primate = data['mg'], data['mg_dev'], data['primate']
    w = mg['work'][(16, 2)]
    rows = []
    if mg['n']:
        status = 'complete' if mg['n'] == 30 else f"partial: {mg['n']}/30 repeats"
        rows.append([f"Ours: race member, mix8 (primary, {status})", f"{mg['modes']['mix8']:.2f}", '57,572 (float32)',
                     f"{w['mix8_macs']:,.0f} (traced)"])
        for m in ('argmax', 'sampled'):
            rows.append([f"Ours: same weights, {m} (reporting only)", f"{mg['modes'][m]:.2f}", '57,572',
                         f"{w['winner_stream_macs']:,.0f} (traced)"])
    for name, smape, foot, macs in MG_REFERENCES:
        rows.append([name, f"{smape:.2f}", f"{foot:,.0f}", f"{macs:,.0f} (effective MACs)"])
    dev_rows = [[d['label'], d['member'], ', '.join(f"{k} {v:.1f}" for k, v in d['modes'].items()), str(d['repeats'])]
                for d in dev]
    pr_rows = [[p['label'], f"{p['test']:.3f}", f"{p['val']:.3f}" + (' (4 blocks)' if p['val_blocks'] > 1 else ''),
                f"{p['params']:,}"] for p in primate]
    pr_rows += [[f"{k} (published, this session)", f"{v:.3f}", '—', '—'] for k, v in PRIMATE_DEV_SESSION.items()]
    lb_rows = [[n, f"{r2:.3f}", f"{foot:,}"] for n, r2, foot in PRIMATE_LEADERBOARD]
    spread = f"; per-repeat mix8 range {mg['spread'][0]:.1f}–{mg['spread'][1]:.1f}" if mg['spread'] else ''
    return [[('h1', 'Appendix. Public benchmarks: NeuroBench and SHD (status from completed result files)'),
             ('p', 'Targets and rules: experiments/SOTA_TARGETS.md. Official protocols are fixed before they run and use the '
                   'vendored official neurobench 2.3.0 loaders/metrics (slices proven identical by contract tests). Development '
                   'results select settings on other series (Mackey-Glass tau 18/19) or on validation splits (primate).'),
             ('h2', 'NeuroBench chaotic function prediction (Mackey-Glass tau 17, sMAPE, lower is better)'),
             ('table', (['Model', 'sMAPE', 'Footprint (bytes)', 'Inference ops / step'], rows, [70, 22, 36, 44])),
             ('small', f"Official protocol: 30 start offsets, 750 teacher-forced training points, 750 autonomous predictions{spread}. "
                       'Our model has 14,393 parameters; the primary inference mode (averaging 8 race-noise streams) was fixed on '
                       'tau 18 development before tau 17 was scored. Our operation counts come from the operator tracer '
                       '(multiply-adds); NeuroBench counts hooked layer MACs; the conventions differ.'),
             ('h2', 'Mackey-Glass development (tau 18/19, three official-protocol repeats per arm)'),
             ('table', (['Arm', 'Member', 'Mean sMAPE by inference mode', 'Repeats'], dev_rows, [64, 28, 64, 16])),
             ('small', 'Findings (THEORY §§417–418): sampled races inject output noise; deterministic or averaged inference helps; '
                       'closed-loop training did not; the exact-expected-reception member (deterministic delivery, hard writes) is '
                       'under development. Three repeats cannot rank close arms (per-repeat spread is large).')],
            [('h2', 'NeuroBench primate reaching (development session indy_20170131_02, R², higher is better)'),
             ('table', (['Arm', 'Test R²', 'Validation R²', 'Params'], pr_rows, [86, 22, 34, 22])),
             ('p', 'Leaderboard (mean of six official sessions): the six-session protocol run with a configuration fixed on '
                   'validation is prepared on AWS; no leaderboard cell is filled from one development session.'),
             ('table', (['Leaderboard entry', 'Six-session R²', 'Footprint (bytes)'], lb_rows, [70, 40, 40])),
             ('small', 'Validation and test rank the development arms differently; the protocol selects by validation only. '
                       'Test on the development session was observed during development; the six-session report will also give '
                       'the mean of the five untouched sessions.'),
             ('h2', 'SHD and other public benchmarks'),
             ('p', 'SHD (official files, speaker-held-out validation) development runs are queued. The completed public '
                   'time-series campaign (ECG200, JapaneseVowels, PenDigits) is negative and reported in its own appendix.')]]
