"""Non-training controller: replay diagnostics only after admitted fits complete."""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--fit-manifest', required=True)
    parser.add_argument('--analysis-manifest', required=True); args = parser.parse_args()
    fits = json.loads((ROOT / args.fit_manifest).read_text())
    analysis = json.loads((ROOT / args.analysis_manifest).read_text())
    jobs = [j for j in fits['jobs'] if j['driver'] == 'experiments/market/b10_tpp.py']
    if len(jobs) != 3:
        raise ValueError('Exactly three admitted B10 development jobs required')
    state_dir = ROOT / '.git/curie-recovery-preview' / analysis['tag']; state_dir.mkdir(parents=True, exist_ok=True)

    def save(state):
        state['observed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        temporary = state_dir / 'await.tmp'; temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_dir / 'await.json')

    while True:
        missing = [j['result'] for j in jobs if not (ROOT / j['result']).exists()]
        if not missing:
            save(dict(status='fits_available', fits=[j['result'] for j in jobs]))
            subprocess.run([str(ROOT / '.venv-docker/bin/python'), 'scripts/run_curie_recovery.py',
                            '--manifest', args.analysis_manifest], cwd=ROOT, check=True)
            save(dict(status='completed', analysis=[j['result'] for j in analysis['jobs']]))
            return
        parent_state = ROOT / '.git/curie-recovery-preview' / fits['tag'] / 'state.json'
        if parent_state.exists():
            state = json.loads(parent_state.read_text())
            if state['status'] in ('needs_review', 'completed'):
                save(dict(status='upstream_needs_review', missing=missing, upstream=state)); return
        save(dict(status='awaiting_completed_fits', missing=missing))
        time.sleep(30)


if __name__ == '__main__':
    main()
