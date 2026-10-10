"""Curie-only continuation: preserve PAM, then admitted numerical/scoring/DEV work.

No fitting is launched directly: every phase uses the pinned one-job queues
through run_curie_recovery/run_safe. TEST reservations remain once-only; any
failed phase stops for review. No AWS access, messages or job reassignment.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def publish(manifest_path):
    manifest=json.loads((ROOT/manifest_path).read_text());paths=[]
    for job in manifest['jobs']:
        result=ROOT/job['result']
        record=json.loads(result.read_text())
        if record.get('status')!='completed' or record.get('tag')!=job['tag']:
            raise ValueError('Completed phase result identity differs')
        if not job['result'].startswith('.git/'):
            paths.append(job['result'])
        score=record.get('per_run_scores',{})
        if score.get('path'):paths.append(score['path'])
    if manifest.get('battle')=='B3':
        paths.append('experiments/results/fas/fas_v2_test_ledger.jsonl')
    paths=[p for p in dict.fromkeys(paths) if (ROOT/p).exists()]
    if not paths:
        return  # Private report receipts have no repository evidence to publish.
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='main':
            raise ValueError('Evidence publication must be on main')
        subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=ROOT).returncode:
            subprocess.run(['git','commit','--only','-m','Publish completed Curie '+manifest.get('battle','B2/B10')+' phase','--',*paths],cwd=ROOT,check=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);a=ap.parse_args()
    if os.uname().nodename!='curie':raise ValueError('Curie owns this continuation')
    manifest=json.loads((ROOT/a.manifest).read_text())
    directory=ROOT/'.git/curie-continuation-preview'/manifest['tag'];directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='waiting_for_pam',completed=[],started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    def save():
        tmp=directory/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(directory/'state.json')
    save();deadline=time.monotonic()+86400
    try:
        while not (ROOT/manifest['pam_result']).exists():
            if time.monotonic()>deadline:raise TimeoutError('PAM completion did not arrive within 24 h')
            time.sleep(30)
        pam=json.loads((ROOT/manifest['pam_result']).read_text())
        if pam.get('status')!='completed':raise ValueError('PAM result incomplete')
        for phase in manifest['phases']:
            state.update(status='waiting_for_memory',phase=phase['name']);save()
            while True:
                available=next(int(x.split()[1])//1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
                if available>=12288:break
                if time.monotonic()>deadline:raise TimeoutError('Continuation headroom did not become available')
                time.sleep(30)
            state.update(status='running',phase=phase['name']);save()
            subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/run_curie_recovery.py','--manifest',phase['manifest']],
                           cwd=ROOT,env=dict(os.environ,MIN_AVAIL_MB='9216'),check=True)
            publish(phase['manifest'])
            state['completed'].append(phase['name']);save()
        state.update(status='completed',phase=None);save()
    except Exception as error:
        state.update(status='needs_review',error=str(error));save();raise


if __name__=='__main__':main()
