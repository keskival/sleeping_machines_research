"""B3 queued synthetic scoring/decision contracts. No FAS data or fitting."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/fas'))
from sealed_registry import atomic_json, sha


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag',required=True); args=parser.parse_args()
    output=ROOT/'experiments/results/fas'/f'{args.tag}.json'
    if output.exists(): raise FileExistsError(output)
    start=time.monotonic()
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_fas_sealed_scoring.py'],cwd=ROOT,check=True)
    paths=('experiments/fas/check_sealed_tools_v2.py','experiments/fas/score_sealed_v2.py',
           'experiments/fas/sealed_registry.py','experiments/fas/stage5_statistics_v2.py',
           'experiments/fas/stage5_decision_v2.py','tests/test_fas_sealed_scoring.py',
           'scripts/await_curie_fas_stage4.py')
    record=dict(status='completed',battle='B3',tag=args.tag,training=False,
                decision='Admit repaired sealed scoring only after interruption, pairing and source contracts pass',
                tests=8,scope='Synthetic temporary files and score arrays only; no model fit or FAS test scoring',
                source_sha256={path:sha(ROOT/path) for path in paths},wall_s=time.monotonic()-start)
    atomic_json(output,record);print(json.dumps(record),flush=True)


if __name__ == '__main__':main()
