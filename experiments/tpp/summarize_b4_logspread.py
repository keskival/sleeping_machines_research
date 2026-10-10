"""B4 result-only decision on recovered TRAIN updates, never a test verdict."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True)
    p.add_argument('--pilot', required=True)
    a = p.parse_args()
    path = ROOT/a.pilot
    r = json.loads(path.read_text())
    assert r['status'] == 'completed' and r['stage'] == 'logspread_development_pilot'
    assert 'test' not in r
    for source, digest in r['source_sha256'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
    passed = r['epochs_run'] == 5 and r['skipped_updates'] == 0
    result = dict(status='completed', tag=a.tag, battle='B4 development',
                  stage='logspread_update_decision', recovered_update_gate=passed,
                  epochs_run=r['epochs_run'], skipped_updates=r['skipped_updates'],
                  val=r['val'], wall_s=r['wall_s'],
                  decision='Recovered updates in five-pass pilot; next requires completed development quality fit.' if passed else
                           'Pilot fails update gate; diagnose its skipped batches or incomplete passes before scaling.',
                  scope='One fresh seed, five-pass TRAIN/DEV numerical development; frozen B4 benchmark unchanged.',
                  input=dict(path=a.pilot, sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                  source_sha256={'experiments/tpp/summarize_b4_logspread.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    out = ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json'
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
