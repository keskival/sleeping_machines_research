"""Wait for source-bound depth phase publication, then safe report refresh."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    state=ROOT/'.git/curie-continuation-preview/curie_trained_write_v6_20261010T1920Z_controller/state.json';deadline=time.monotonic()+86400
    while True:
        d=json.loads(state.read_text()) if state.exists() else {}
        if d.get('status')=='completed':break
        if d.get('status')=='needs_review':raise RuntimeError(d)
        if time.monotonic()>deadline:raise TimeoutError('Trained-write phase did not complete')
        time.sleep(30)
    subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/run_curie_recovery.py','--manifest','experiments/queue/enabler/curie_trained_write_report_20261010T1930Z/manifest.json'],cwd=ROOT,check=True)
if __name__=='__main__':main()
