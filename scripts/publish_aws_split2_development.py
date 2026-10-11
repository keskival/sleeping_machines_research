"""Finite automatic publication of the completed split2 quality/work decision."""
import fcntl,json,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PYTHON=str(ROOT/'.venv-docker/bin/python');TAG='aws_b4_split2_development_decision_20261011';STATE=ROOT/'.git'/f'{TAG}.publication.json'
def run(args):return subprocess.run(args,cwd=ROOT,check=True)
def output(args):return subprocess.check_output(args,cwd=ROOT,text=True)
def state(status):STATE.write_text(json.dumps(dict(status=status,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
def main():
 deadline=time.monotonic()+12000;path=ROOT/'experiments/results/tpp_b4'/f'{TAG}.json';state('waiting for completed full-fit decision')
 while time.monotonic()<deadline:
  if not path.exists():time.sleep(20);continue
  d=json.loads(path.read_text());assert d['status']=='completed'
  with open('/tmp/aws-language-publication.lock','a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if output(['git','status','--porcelain','--untracked-files=no']).strip():state('waiting for clean tracked tree');time.sleep(2);continue
   assert output(['git','branch','--show-current']).strip()=='main';run(['git','pull','--rebase'])
   sys.path.insert(0,str(ROOT/'report'));from b4_mark_development_appendix import split2_development_blocks
   assert split2_development_blocks()
   if d['repaired_fit_completed']:
    text=f"Full Github split2 seed0 repaired fit completes {d['epochs']} passes, zero skips; selected DEV total {d['selected_repaired_dev']['total']:.3f} vs guarded {d['guarded_dev']['total']:.3f}, gain {d['selected_total_gain']:.3f}. Registered selected-gain>=0.5 gate {'passes' if d['quality_gate'] else 'fails'}. Fitting {d['fitting_wall_s']:.2f}s, DEV evaluation {d['evaluation_wall_s']:.2f}s, paid pilot {d['paid_pilot_wall_s']:.2f}s; peak RSS {d['fit_max_rss_kb']/1024:.1f}MiB. Same Github cohort/new partition, one seed; no new TEST or benchmark replacement. Next mark/readout development uses this completed decision; no mark fit is automatically admitted."
   else:text='Full split2 repaired fit fails its finite/no-skip completion gate. Current anchored compact readout and frozen benchmark scores retained; diagnose the completed fit before further development.'
   handoff=ROOT/'experiments/HANDOFF.md';parts=handoff.read_text().split('\n\n');header='**AWS prospective split2 pilot passes; full repaired fit admitted'
   parts=[p for p in parts if not p.startswith(header)];handoff.write_text('**AWS split2 full-fit quality/work decision published.** '+text+'\n\n'+'\n\n'.join(parts))
   doc=ROOT/'experiments/tpp/SINGLE_MARK_READOUT_DEVELOPMENT.md';doc.write_text(doc.read_text()+'\n## Current split2 full-fit decision\n\n'+text+'\n')
   science=ROOT/'report/I_SCIENCE.md';science.write_text(science.read_text().replace('its full original-budget repaired fit is running.','its full original-budget repaired fit '+('completes with selected DEV total gain '+f"{d['selected_total_gain']:.3f}." if d['repaired_fit_completed'] else 'fails the finite completion gate.')))
   for script in ['report/make_pdf.py','report/split_report.py']:run([PYTHON,script])
   files=output(['git','diff','--name-only']).splitlines();allowed={'REPORT.md','experiments/HANDOFF.md','experiments/tpp/SINGLE_MARK_READOUT_DEVELOPMENT.md','report/I_SCIENCE.md','report/II_METHODS.md','report/III_RECORD.md'}
   assert all(f in allowed or (f.startswith('report/') and f.endswith('.pdf')) or f.startswith('report/figures/') for f in files)
   run(['git','diff','--check']);run(['git','add','--',*files]);run(['git','commit','-m','Publish completed prospective split2 development quality and work','--only','--',*files]);pushed=subprocess.run(['git','push'],cwd=ROOT)
   state('completed and pushed' if pushed.returncode==0 else 'local completed report committed; remote synchronization pending');return
 raise TimeoutError('Bounded full-fit publication deadline')
if __name__=='__main__':
 try:main()
 except Exception as e:state('stopped: '+repr(e));raise
