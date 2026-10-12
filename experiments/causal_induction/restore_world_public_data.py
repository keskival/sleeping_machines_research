"""Restore only admitted public systems, with byte-exact original identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nonlinear_generate import episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a = p.parse_args()
    parent = ROOT/'experiments/results/causal_induction/curie_causal_nonlinear_20261012_v2_prepare.json'
    data = json.loads(parent.read_text()); assert data['status'] == 'completed'
    for name, digest in data['source_sha256'].items(): assert sha(ROOT/name) == digest, name
    selected = [r for r in data['records'] if r['split'] == 'train' and r['variant'] == 'base']
    for split in ('heldout_graph', 'heldout_factors', 'heldout_depth', 'heldout_factors_depth'):
        selected += [next(r for r in data['records'] if r['split'] == split and r['variant'] == 'base')]
        selected += [[r for r in data['records'] if r['split'] == split and r['variant'] == 'rewire'][1]]
    assert len(selected) == 24
    restored = []
    for row in selected:
        path = ROOT/row['path']
        assert path.resolve().is_relative_to(ROOT/'data/causal_induction')
        if path.exists():
            assert sha(path) == row['sha256'], path
            restored.append(dict(path=row['path'], sha256=row['sha256'], restored=False)); continue
        match = re.fullmatch(r'.+_(\d+)_(base|rewire)\.npz', path.name); assert match
        public, _ = episode(int(match[1]), row['depth'], row['width'], switch=match[2] == 'rewire')
        # Oracle metadata is discarded; never written or passed to a learner.
        assert set(public) == {'events', 'queries', 'feedback', 'capacity'}
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pending', delete=False) as f:
            pending = Path(f.name); np.savez(f, **public)
        try:
            assert sha(pending) == row['sha256'], f'Generator does not reproduce {path}'
            try: os.link(pending, path)
            except FileExistsError: assert sha(path) == row['sha256']
        finally: pending.unlink()
        restored.append(dict(path=row['path'], sha256=sha(path), restored=True))
    source = str(Path(__file__).relative_to(ROOT))
    result = dict(status='completed', battle='B5/R1', records=restored,
                  input_sha256={str(parent.relative_to(ROOT)):sha(parent)},
                  source_sha256={source:sha(Path(__file__)), **data['source_sha256']},
                  scope='24 previously admitted public systems restored only after exact original SHA-256 match. '
                        'No oracle files, new data selection, fitting, scoring or changed protocol. Existing files verified without overwrite.')
    out = ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(dict(systems=len(restored), restored=sum(r['restored'] for r in restored))), flush=True)


if __name__ == '__main__': main()
