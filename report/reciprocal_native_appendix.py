"""Current native reciprocal-credit development from completed source-bound data."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TITLE='Appendix. R1 reciprocal learning: native gates and credit transfer development'


def pages():
    prefix='curie_credit_versioned_20261010T1028Z'
    folder=ROOT/'experiments/results/credit'
    contract=folder/(prefix+'_contract_d2.json')
    pilot=folder/'curie_credit_transfer_v1_20261010T1033Z_pilot.json'
    if not contract.exists() or not pilot.exists():return []
    c=json.loads(contract.read_text());p=json.loads(pilot.read_text())
    if c['status']!='completed' or p['status']!='completed':raise ValueError('Completed native evidence required')
    gates=c['metrics'];rows=[]
    for depth in (2,4,8):
        s=json.loads((folder/(prefix+f'_smoke_d{depth}.json')).read_text())
        ledger=s['metrics']['ledger']
        unique=ledger['actor_local_scored_targets']+ledger['future_meta_scored_targets']
        work=s['supported_flops']
        rows.append([str(depth),str(ledger['actor_update_parameter_visits']//3),str(unique),
                     f'{work/1e6:.3f}',f'{work/unique/1e6:.4f}',
                     f'{s["wall_s"]:.2f}',f'{s["peak_rss_kb"]/1024:.1f}'])
    transfer=[]
    for r in p['metrics']['checkpoints'][-1]['assessment']:
        transfer.append([r['case'],str(r['depth']),str(r['pairs']),f'{r["relative_missing_credit_mse"]:.4f}',
                         f'{r["frozen_relative_mse"]:.4f}'])
    # The v5 contract is selected-teacher/state parity; exact one-step
    # derivatives are certified by the preserved v1 contract, separately.
    old=json.loads((folder/'curie_reciprocal_native_v1_20261010T1008Z_contract.json').read_text())
    derivative=max(v['absolute_error'] for r in old['metrics'] for v in r['one_step_future_meta_derivatives'].values())
    return [[('h1',TITLE),
             ('p','R1/B1 development: the unchanged native keyed temporal model supports persistent credit functions '
              'and a learned update function. Depths 2, 4 and 8 pass selected-teacher and temporal-state parity; '
              f'one-step future derivatives for credit and optimizer parameters agree with finite differences to {derivative:.2g}. '
              'Task-preserving key/value relabeling respects forward predictions, gradients and learned updates. '
              'Version-bound packets repair stale embedding features exposed by that symmetry test.'),
             ('table',(['Depth','Actor params','FIT targets','Whole-fit MF','MF/target','Wall s','RSS MiB'],rows,[15,24,23,29,27,20,24])),
             ('small','Three updates per depth; unique FIT targets include support and independent future examples. '
              'Whole-fit/per-target arithmetic use the same PyTorch supported-operation convention; special functions and '
              'unsupported operations remain in the saved operator ledger and are not represented fully by this FLOP number. '
              'Wall includes instrumentation. Every native model has eight addressed mark slots and eight key slots, '
              'writes one of each per input, updates every temporal layer and scores all eight keys. Local actor backward, '
              'predicted coordinate updates and optimizer visits remain dense; selected teachers query only audited producer '
              'blocks. Inference is unchanged and not separately timed by these gates. No efficiency, benchmark or energy claim.'),
             ('p','The frozen-teacher coordinate-credit prototype did not produce useful calibration transfer after its '
              'fixed 64-step pilot. The actor snapshots stay fixed, all assessment credit states reset, and no assessment '
              'labels enter predictions. Relative missing-credit MSE 1.0 is the zero-correction baseline. '
              'This diagnoses the coarse coordinate feature interface; it is not a verdict on the model family or grokking.'),
             ('table',(['DEV case','Depth','Pairs','Learned MSE','Frozen MSE'],transfer,[48,20,20,34,34])),
             ('small',f'Pilot: {p["metrics"]["train_sequences"]} TRAIN sequences, disjoint DEV seeds, no TEST; '
              f'{p["metrics"]["original_scored_targets"]} scored-target presentations with '
              f'{p["metrics"]["teacher_gradient_targets"]} full/local teacher-gradient targets, '
              f'{p["supported_flops"]/1e6:.2f} supported MFLOPs, {p["wall_s"]:.2f} s, '
              f'{p["peak_rss_kb"]/1024:.1f} MiB peak RSS. All teacher and assessment work included.'),
             ('p','Next gate: state-sized addressed key cotangents with exact forward elapsed-time transport, then learned '
              'credit tails and asynchronous credit clocks. Causal streams use actual timestamps and versioned pending '
              'records. Credit structure and optimizer parameters may adapt across regimes; frozen teachers isolate '
              'calibration capacity and do not assume stationary deployment. See CREDIT_GENERALIZATION.md and '
              'RECIPROCAL_EVENT_LEARNING.md. No delayed-generalization/grokking result is claimed.')]]
