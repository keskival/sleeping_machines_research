"""Publish completed B4 calibration/serving evidence under the host publication lock.
No model execution. Waiting/failed/push-pending states stay in .git, not report history.
"""
import fcntl,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TAG='aws_b4_calibration_confirmation_20261011'
STATE=ROOT/'.git'/f'{TAG}.publication.json';PYTHON=str(ROOT/'.venv-docker/bin/python')
def run(args,**kw):return subprocess.run(args,cwd=ROOT,check=True,**kw)
def state(status):STATE.write_text(json.dumps(dict(status=status,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
def output(args):return subprocess.check_output(args,cwd=ROOT,text=True)
def main():
    deadline=time.monotonic()+24000;tags=[TAG+'_decision']+[TAG+f'_serving_s{s}' for s in range(3)]
    state('waiting for completed decision and serving diagnostics')
    while time.monotonic()<deadline:
        time.sleep(10)
        paths=[ROOT/'experiments/results/tpp_b4'/f'{tag}.json' for tag in tags]
        if not all(p.exists() for p in paths):time.sleep(30);continue
        if not all(json.loads(p.read_text())['status']=='completed' for p in paths):raise RuntimeError('failed publication prerequisite')
        with open('/tmp/aws-language-publication.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if output(['git','status','--porcelain','--untracked-files=no']).strip():state('waiting for clean tracked tree');continue
            assert output(['git','branch','--show-current']).strip()=='main'
            run(['git','pull','--rebase'])
            import sys
            sys.path.insert(0,str(ROOT/'report'))
            from b4_mark_development_appendix import confirmation_blocks
            assert confirmation_blocks()
            d=json.loads(paths[0].read_text());verdict='passes' if d['gate'] else 'fails'
            text=f"Fixed-recipe three-seed calibration gate {verdict}: mean DEV mark gain {d['mean_mark_gain']:.3f}, SD {d['sample_sd']:.3f}; mean>=1 and eachseed>0. One Github split1, seed0 selected and1/2 prospectively tested recipe, repaired timing fixed, both predictor fits/serving charged. No TEST, benchmark/efficiency claim or failed-gate relaxation. Decision and three serving/error diagnostics completed; next single-readout engineering follows these results."
            doc=ROOT/'experiments/tpp/B4_MARK_MEMORY_DEVELOPMENT.md';heading='\n## Current calibration confirmation result\n'
            doc.write_text(doc.read_text().split(heading)[0]+heading+'\n'+text+'\n')
            handoff=ROOT/'experiments/HANDOFF.md';handoff.write_text('**AWS calibration confirmation and serving evidence published.** '+text+'\n\n'+handoff.read_text())
            for script in ['report/make_pdf.py','report/split_report.py']:run([PYTHON,script])
            files=output(['git','diff','--name-only']).splitlines()
            allowed={'REPORT.md','experiments/HANDOFF.md','experiments/tpp/B4_MARK_MEMORY_DEVELOPMENT.md','report/II_METHODS.md','report/III_RECORD.md'}
            if any(f not in allowed and not (f.startswith('report/') and f.endswith('.pdf')) and not f.startswith('report/figures/') for f in files):raise RuntimeError('unexpected tracked change during publication')
            run(['git','diff','--check']);run(['git','add','--',*files]);run(['git','commit','-m','Publish fixed-recipe calibration confirmation and serving costs','--only','--',*files])
            pushed=subprocess.run(['git','push'],cwd=ROOT)
            state('completed and pushed' if pushed.returncode==0 else 'local report committed; remote sync pending concurrent update')
            return
    raise TimeoutError('bounded publication waiting deadline')
if __name__=='__main__':
    try:main()
    except Exception as e:state('stopped: '+repr(e));raise
