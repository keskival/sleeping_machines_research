"""Finite locked publication of completed split2 mark stability/quality evidence."""
import fcntl,json,time,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TAG='aws_b4_split2_marks_decision_20261011';STATE=ROOT/'.git'/f'{TAG}.publication.json';PYTHON=str(ROOT/'.venv-docker/bin/python')
def run(args):return subprocess.run(args,cwd=ROOT,check=True)
def out(args):return subprocess.check_output(args,cwd=ROOT,text=True)
def state(s):STATE.write_text(json.dumps(dict(status=s,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
def main():
 start=time.monotonic();path=ROOT/'experiments/results/tpp_b4'/f'{TAG}.json';state('waiting for completed mark-fit decision')
 while time.monotonic()-start<12000:
  if not path.exists():time.sleep(20);continue
  d=json.loads(path.read_text());assert d['status']=='completed'
  with open('/tmp/aws-language-publication.lock','a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if out(['git','status','--porcelain','--untracked-files=no']).strip():time.sleep(2);continue
   assert out(['git','branch','--show-current']).strip()=='main';run(['git','pull','--rebase']);sys.path.insert(0,str(ROOT/'report'));from b4_mark_development_appendix import split2_marks_blocks
   assert split2_marks_blocks()
   if d['fits_completed_without_skips']:text=f"Split2 seed0 mark fit completed without skips; registered last20-common mark gate {'passes' if d['gate'] else 'fails'} at {d['last20_common_mark_gain']:.3f} vs required1.000. Selected mark gain {d['selected_mark_gain']:.3f}, total gain {d['selected_total_gain']:.3f}. Same Github cohort/new partition, original budget; no TEST or benchmark replacement. Core/mark/pilot/validation work charged. Next TRAIN-only calibration/readout decision needs its own registration; failed stability gate is never relaxed."
   else:text='Split2 mark fit fails its finite/no-skip completion gate. Preserve completed evidence; retain repaired core and current compact readout. Diagnose the failure before another fit; no TEST or family verdict.'
   handoff=ROOT/'experiments/HANDOFF.md';handoff.write_text('**AWS split2 mark quality/work decision published.** '+text+'\n\n'+handoff.read_text());doc=ROOT/'experiments/tpp/SINGLE_MARK_READOUT_DEVELOPMENT.md';doc.write_text(doc.read_text()+'\n## Current split2 mark-fit decision\n\n'+text+'\n')
   for script in ['report/make_pdf.py','report/split_report.py']:run([PYTHON,script])
   files=out(['git','diff','--name-only']).splitlines();allowed={'REPORT.md','experiments/HANDOFF.md','experiments/tpp/SINGLE_MARK_READOUT_DEVELOPMENT.md','report/II_METHODS.md','report/III_RECORD.md'}
   assert all(f in allowed or (f.startswith('report/') and f.endswith('.pdf')) or f.startswith('report/figures/') for f in files)
   run(['git','diff','--check']);run(['git','add','--',*files]);run(['git','commit','-m','Publish prospective split2 mark quality and complete work','--only','--',*files]);pushed=subprocess.run(['git','push'],cwd=ROOT);state('completed and pushed' if pushed.returncode==0 else 'committed current report; remote synchronization pending');return
 raise TimeoutError('Bounded mark publication deadline')
if __name__=='__main__':
 try:main()
 except Exception as e:state('stopped: '+repr(e));raise
