"""Rebuild current report artifacts; publication receipt stays under .git."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def require_main():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    conflicts=subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True).strip()
    if branch!='main' or conflicts or any((ROOT/'.git'/name).exists() for name in ['rebase-merge','rebase-apply']):
        raise ValueError('Report publication requires main without a concurrent rebase or unresolved files')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=ROOT/'.git/curie-report-build-preview'/a.tag/'result.json';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    start=time.monotonic()
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        require_main()
        for script in ['report/make_pdf.py','report/split_report.py']:
            subprocess.run([str(ROOT/'.venv-docker/bin/python'),script],cwd=ROOT,check=True)
        require_main()
        paths=['report/figures/current_native_language_status.svg','report/figures/latest_native_language_fitting.svg','report/sleeping_machines_status.family.text.pdf','REPORT.md','report/sleeping_machines_status.pdf','report/II_METHODS.md','report/III_RECORD.md']
        paths.extend(str(p.relative_to(ROOT)) for p in (ROOT/'report').glob('I*_*.pdf'))
        paths=sorted(set(p for p in paths if (ROOT/p).exists()))
        for attempt in range(15):
            require_main()
            added=subprocess.run(['git','add','--',*paths],cwd=ROOT,text=True,capture_output=True)
            if added.returncode==0:break
            if 'index.lock' not in added.stderr:raise RuntimeError(added.stderr)
            time.sleep(2)
        else:raise RuntimeError('Git index stayed busy during report publication')
        if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=ROOT).returncode:
            require_main()
            subprocess.run(['git','commit','--only','-m','Regenerate current report with native reciprocal credit development','--',*paths],cwd=ROOT,check=True)
        require_main()
        sources=['report/online_episode_appendix.py','report/I_SCIENCE.md','scripts/build_curie_episode_report.py','report/reciprocal_native_appendix.py',
                 'report/readable_report.py','report/make_pdf.py','report/split_report.py']
        result=dict(status='completed',tag=a.tag,battle='R1/B1',training=False,wall_s=time.monotonic()-start,
                    publication_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                    artifact_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
        out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
