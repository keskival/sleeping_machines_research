"""B4 development five-pass TRAIN/DEV pilot, gated by full backward contract."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import race_tpp_b4g as adapter
from race_tpp_logspread import RaceTPP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--gate', required=True)
    args = parser.parse_args()
    gate_path = adapter.ROOT/args.gate
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'completed'
    assert gate['captured']['head_gradients_finite'] and gate['captured']['full_gradients_finite']
    for source, digest in gate['source_sha256'].items():
        assert hashlib.sha256((adapter.ROOT/source).read_bytes()).hexdigest() == digest, source
    # Fresh original seed/settings; only the numerical representation changes.
    # Five passes include epoch four, where the saved original first skipped.
    adapter.base.RaceTPP = RaceTPP
    sys.argv = [sys.argv[0], '--tag', args.tag, '--dataset', 'github_filtered',
                '--split', '2', '--seed', '0', '--epochs', '5', '--max-wall-s', '1800']
    adapter.main()
    path = adapter.ROOT/'experiments/results/tpp_b4'/f'{args.tag}.json'
    result = json.loads(path.read_text())
    assert 'test' not in result
    result.update(stage='logspread_development_pilot', scope='Five-pass fresh Github split2 seed-0 TRAIN/DEV numerical-repair pilot; no TEST and no frozen benchmark replacement.',
                  gate=dict(path=args.gate, sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest()))
    sources = ['experiments/tpp/b4_logspread_split2_pilot.py', 'experiments/tpp/race_tpp_logspread.py',
               'experiments/tpp/race_tpp_b4g.py', 'experiments/tpp/race_tpp_v19.py']
    result['source_sha256'] = {s:hashlib.sha256((adapter.ROOT/s).read_bytes()).hexdigest() for s in sources}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(path)


if __name__ == '__main__':
    main()
