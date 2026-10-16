"""Source-bound RMR training with exact zero-old-write execution elimination.

Uses the established acquisition, optimizer, checkpoint and scoring protocol.
The numerical gate is required by the coordinator admission, not bypassed here.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments')]
from experiments import r1_rmr
from experiments.r1_rmr_compact import CompactRMR


def main():
    assert '--top-m' in sys.argv and sys.argv[sys.argv.index('--top-m')+1] == '0'
    assert '--mode' not in sys.argv or sys.argv[sys.argv.index('--mode')+1] == 'train'
    tag = sys.argv[sys.argv.index('--tag')+1]
    files = ['experiments/r1_rmr_compact_train.py', 'experiments/r1_rmr_compact.py',
             'experiments/r1_rmr.py', 'experiments/r1_cerl.py',
             'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py']
    sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    r1_rmr.RMR = CompactRMR
    r1_rmr.main()
    path = ROOT/f'experiments/results/token_language/{tag}.json'
    if not path.exists():
        return  # established bounded max-steps checkpoint mode
    result = json.loads(path.read_text())
    result['source_sha256'] = sources
    result['execution'] = dict(old_slot_write_matrices='Absent because every top_m=0; original prediction law and trained parameters retained.',
                               scope='Carrier-backed residual-memory development member; lane-bounded numerical state and BPTT remain. No claim of a fully native event-local learner.')
    result['training_work_scope'] = 'Saved train_flops fields are registered main-NLL forward/backward subtotals. Exploration backward, optimizer, unregistered special/scalar operations and traffic are not included; no complete-work or hardware claim.'
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(result, indent=2)+'\n')
    os.replace(tmp, path)


if __name__ == '__main__':
    main()
