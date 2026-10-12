"""R1 memory-input contract; execute through the guarded battle queue."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True)
    args = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_route_activation_memory.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)
    names = ['sleeping_machines/route_activation_memory.py', 'tests/test_route_activation_memory.py',
             'experiments/credit/check_route_activation_memory.py']
    packet = dict(status='completed', battle='R1', checks=result.testsRun,
                  source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
                  scope='Exact weighted-mixture moments, issued-copy causality, version freshness and bounded admission. '
                        'Unintegrated observation-memory component; no learned route geometry or task-quality claim.')
    path = ROOT/f'experiments/results/diagnostics/{args.tag}.json'
    with path.open('x') as f: json.dump(packet, f, indent=2); f.write('\n')
