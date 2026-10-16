"""Await one integrated native acquisition,publish completed evidence once,stop."""
import fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TAG='aws_g_hier_identity_role_20261016_v1'
RESULT=ROOT/f'experiments/results/hierarchy/{TAG}.json';STATE=ROOT/'.git/identity_role_publication_20261016.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**fields):
 p=STATE.with_suffix('.writing');p.write_text(json.dumps(fields,indent=2)+'\n');p.replace(STATE)
def publish(r):
 assert r['status']=='completed' and r['stage']=='identity_role_learning'
 baseline_name='experiments/results/hierarchy/aws_g_hier_grounded_decoder_only_20261016_v1.json'
 old=json.loads((ROOT/baseline_name).read_text());reference=old['final']['materialized_dev_nll']
 nll=r['final']['materialized_dev_nll'];accuracy=r['final']['shape_dev']['accuracy'];initial=r['initial']['shape_dev']['accuracy']
 interpretation='Whole-label development loss improves the saved acquired reference.' if nll<reference else 'Shape learning and whole-label transfer remain separate;the saved acquired actor retains the better whole-label result.'
 paragraph=f"**Completed identity-preserving role acquisition.** Parent-used issued-view shape accuracy on reused DEV changes{100*initial:.2f}%→{100*accuracy:.2f}%;whole-label NLL{nll:.6f} versus saved acquired{reference:.6f}. {interpretation} One integrated8level/32stage actor,64paired TRAIN updates,TRAIN-only shape targets on actual parent-used versions/ridge.01root head,new combinations/nuisance tapes,TEST excluded. No ablation/control acquisition or public benchmark claim. Source:`investment/identity_role_development_evidence.json`."
 packet=dict(status='completed',battle='G-HIER',dev_nll=nll,reference_dev_nll=reference,initial_shape_dev_accuracy=initial,shape_dev_accuracy=accuracy,initial=r['initial'],final=r['final'],work_training=r['work_training'],scope=r['scope'],decision=interpretation,input_sha256={str(RESULT.relative_to(ROOT)):sha(RESULT),baseline_name:sha(ROOT/baseline_name)})
 files=[]
 def write(name,s):(ROOT/name).write_text(s);files.append(name)
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert not (ROOT/'.git/MERGE_HEAD').exists()
  write('investment/identity_role_development_evidence.json',json.dumps(packet,indent=2)+'\n')
  for name in ['experiments/HANDOFF.md','experiments/hierarchy/CURRENT_FINDINGS.md']:
   s=(ROOT/name).read_text();at=s.index('\n\n');write(name,s[:at]+'\n\n'+paragraph+s[at:])
  for name in ['investment/PITCH.md','investment/INVESTMENT_CASE.md','investment/INVESTOR_PROOF_PLAN.md']:
   s=(ROOT/name).read_text();at=s.index('\n\n');write(name,s[:at]+'\n\n'+paragraph+s[at:])
  name='REPORT.md';write(name,(ROOT/name).read_text()+'\n\n### Native identity-preserving role learning\n\n'+paragraph+'\n\nFull work,identity loss history,TRAIN-only head solver residuals,parameters,presentations and checkpoint bindings are in the completed result. This is a within-family development point;the actor uses the corrected static ownership policy and existing native clocks/role interactions.\n')
  name='report/I_SCIENCE.md';s=(ROOT/name).read_text();at=s.index('\n## 5.');write(name,s[:at]+'\n\n'+paragraph+'\n'+s[at:])
  name='investment/brief/evidence_brief.tex';s=(ROOT/name).read_text();a=s.index('{\\small \\textbf{Native part-identity learning.}');b=s.index('\\begin{center}',a)
  text=f"\\textbf{{Identity-preserving composition.}} The 25\\% support defect is repaired: 8 complete part memories,19 combined checks. After64paired TRAIN updates,issued-view DEV shape accuracy changes {100*initial:.2f}\\% to {100*accuracy:.2f}\\%;whole-label NLL {nll:.6f} versus acquired {reference:.6f}. One8level/32stage actor,TRAIN-only targets,new combinations/nuisances,no TEST or ablation arms. Shape recognition and whole-label transfer are distinct endpoints. Source: \\texttt{{investment/identity\\_role\\_development\\_evidence.json}}.\n\n"
  write(name,s[:a]+'{\\small '+text+'}\n\n'+s[b:])
  scratch=ROOT/'.git/evidence-brief-preview/20261016IdentityRole/brief';scratch.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(ROOT/'investment/brief',scratch,dirs_exist_ok=True)
  env=os.environ.copy();local=ROOT/'experiments/.texmf_local';env.update(TEXMFHOME=str(local),TEXMFVAR=str(local/'var'),TEXMFCONFIG=str(local/'config'))
  with (scratch/'build.log').open('w') as out:subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','evidence_brief.tex'],cwd=scratch,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
  shutil.copy2(scratch/'evidence_brief.pdf',ROOT/'investment/sleeping_machines_evidence_brief.pdf');files.append('investment/sleeping_machines_evidence_brief.pdf')
  for script,args,label in [('report/make_pdf.py',['--current-record'],'report'),('scripts/build_investment_case.py',['--tag','20261016IdentityRole'],'memos')]:
   with (ROOT/f'.git/identity_role_{label}_build.log').open('w') as out:subprocess.run([str(ROOT/'.venv-docker/bin/python'),script,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
  files+=['investment/sleeping_machines_investment_case.pdf','investment/sleeping_machines_investor_pitch.pdf','report/sleeping_machines_status.pdf']+['report/'+n for n in ['I_SCIENCE.pdf','II_METHODS.md','II_METHODS.pdf','III_RECORD.md','III_RECORD.pdf']]
  subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/verify_investor_evidence.py'],cwd=ROOT,check=True)
  subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/package_pitch_diligence.py','--tag','20261016IdentityRole'],cwd=ROOT,check=True);files.append('investment/sleeping_machines_private_diligence.zip')
  subprocess.run(['git','add','--',*files],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish integrated identity-preserving composition result','--',*files],cwd=ROOT,check=True)
  pushed=subprocess.run(['git','push','origin','main'],cwd=ROOT).returncode==0
 status(stage='completed' if pushed else 'completed_local_pending_push',dev_nll=nll,shape_dev_accuracy=accuracy)
def main():
 if STATE.exists() and json.loads(STATE.read_text()).get('stage') in ['completed','completed_local_pending_push','needs_review']:return
 deadline=time.monotonic()+3600
 while time.monotonic()<deadline:
  if RESULT.exists() and (r:=json.loads(RESULT.read_text())).get('status')=='completed':publish(r);return
  status(stage='waiting_completed_integrated_fit');time.sleep(15)
 raise RuntimeError('bounded publication wait expired;inspect exact checkpoints/runner')
if __name__=='__main__':
 try:main()
 except Exception as error:status(stage='needs_review',error=repr(error));raise
