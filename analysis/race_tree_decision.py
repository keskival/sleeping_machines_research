"""R1 saved-evidence decision: race-tree candidate versus acquired keyed members.

No model import, fitting or target scoring. Reports incomplete matched arms as pending
and isolates the tree-only ledger from copy, backbone and optimizer work.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RACE = 'aws_r1_race_tree_L2_1M_s0_20261015_v1'
REFERENCES = ['curie_r1g2_keyed1_1M_s0_20261007T1800Z',
              'curie_r1g2_keyed1_4M_s0_20261008T1140Z']
MATCHED = 'aws_r1_deep_keyed_L2_1M_s0_20261015_v3'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(tag):
    path = ROOT / f'experiments/results/token_language/{tag}.json'
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if d.get('status') != 'completed':
        return None
    score = d['score_65528']
    if score['targets'] != 65528 or not math.isfinite(score['nll']):
        raise ValueError(f'{tag}: incompatible target count or nonfinite score')
    if d['args']['lane'] != 256 or d['args']['seed'] != 0:
        raise ValueError(f'{tag}: unexpected training-lane/seed protocol')
    driver = ('experiments/r1_race_tree_lm.py' if tag == RACE else
              'experiments/r1_deep_keyed_lm.py' if tag == MATCHED else
              'experiments/r1_token_keyed_lm.py')
    source = ROOT/driver
    if d.get('source_sha256', {}).get(driver) != sha(source):
        raise ValueError(f'{tag}: driver changed; recover the bound source before protocol comparison')
    for protocol in ('20971520, 8, 8192', '10485760, 8, 2048'):
        if protocol not in source.read_text():
            raise ValueError(f'{tag}: scoring/selection slice requires manual protocol review')
    return path, d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError('Plain unique tag required')
    found = {tag: read(tag) for tag in [RACE, *REFERENCES, MATCHED]}
    if found[RACE] is None:
        raise ValueError('Race-tree result not complete')
    race_path, race = found[RACE]
    rows, bindings = [], {}
    for tag, pair in found.items():
        if pair is None:
            rows.append(dict(tag=tag, status='pending'))
            continue
        path, d = pair
        bindings[str(path.relative_to(ROOT))] = sha(path)
        rows.append(dict(tag=tag, status='completed', nll=d['score_65528']['nll'],
                         train_unique_tokens=d['args']['train_tokens'], epochs_run=d['epochs_run'],
                         best_epoch=d['best_epoch'], parameters=d['parameters'], process_wall_s=d['wall_s'],
                         gain_of_race_nats=d['score_65528']['nll']-race['score_65528']['nll'],
                         scope='Fixed-weight reused DEV slice, same 65528 targets; architecture/host/acquisition differ unless explicitly paired.'))
    # Audit immutable construction inputs, without treating all historical runtime files
    # as immutable: some references are explicitly versioned by their saved source hashes.
    tree = ROOT/'experiments/results/token_language/vocab_race_tree_b16d4_20261015_v1.npz'
    expected = race['source_sha256'].get(str(tree.relative_to(ROOT)))
    if expected is None or sha(tree) != expected:
        raise ValueError('Race-tree vocabulary artifact changed or lacks binding')
    bindings[str(tree.relative_to(ROOT))] = expected
    matched = found[MATCHED]
    decision = ('Complete the paired keyed-L2 endpoint before assigning backbone/head attribution; '
                'then use saved checkpoints for tree/copy decomposition and same-protocol history interventions. '
                'No new fit or larger race-tree budget is admitted by this packet.')
    out = dict(status='completed', battle='R1', kind='saved_evidence_decision', rows=rows,
               matched_backbone_comparison_ready=matched is not None,
               protocol=dict(validation_shard='data/fineweb_gpt2/fineweb_val_000000.bin',
                   scoring_offset=20971520, scoring_lanes=8, scoring_lane_tokens=8192,
                   selection_offset=10485760, selection_lanes=8, selection_lane_tokens=2048,
                   validation='Current result-bound driver hashes and slice literals checked; target count checked.'),
               race_tree_output_ledger=race['output_work_per_token'],
               accounting_scope='Recorded tree-path child dot products only: not complete normalized-output or whole-model work. '
                   'Excluded: copy key/query projections, all stored-key scores, softmax/log operations, gate, LayerNorm, '
                   'tree construction, backbone, discovery, backward, optimizer, storage and traffic. '
                   'Do not publish the dense/tree MAC ratio as total inference or training savings.',
               mechanism_coverage=dict(temporal_memory=True, keyed_predecessor_reads=True,
                   hierarchical_output=True, causal_copy_mixture=True, full_backprop=True,
                   hard_hidden_receiver_races=False, counterfactual_alternative_state_writes=False,
                   sparse_selected_output_path=True, all_copy_keys_scored=True),
               online_scope='Offline acquired model and fixed-weight DEV score. Not a chronological online-language win. '
                   'Training uses reset 256-token lanes; useful persistence is not proven by this score.',
               decision=decision, input_sha256=bindings)
    output = ROOT/f'experiments/results/diagnostics/{args.tag}.json'
    with output.open('x') as f:
        json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(dict(output=str(output.relative_to(ROOT)), rows=rows, decision=decision), indent=2))


if __name__ == '__main__':
    main()
