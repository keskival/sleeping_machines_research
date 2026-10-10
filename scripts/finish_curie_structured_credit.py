"""Wait for structured pilots, run admitted geometry/policy/report phase and publish."""
import fcntl,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    state=ROOT/'.git/curie-continuation-preview/curie_structured_credit_v5b_20261010T1845Z_controller/state.json';deadline=time.monotonic()+14400
    while True:
        d=json.loads(state.read_text()) if state.exists() else {}
        if d.get('status')=='completed':break
        if d.get('status')=='needs_review':raise RuntimeError(d)
        if time.monotonic()>deadline:raise TimeoutError('Structured phase did not finish')
        time.sleep(20)
    path='experiments/queue/enabler/curie_structured_finish_20261010T1910Z/manifest.json'
    subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/run_curie_recovery.py','--manifest',path],cwd=ROOT,check=True)
    manifest=json.loads((ROOT/path).read_text());paths=[j['result'] for j in manifest['jobs'] if not j['result'].startswith('.git/')]
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
        subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish response geometry and versioned update policy contracts','--',*paths],cwd=ROOT,check=True)
if __name__=='__main__':main()
