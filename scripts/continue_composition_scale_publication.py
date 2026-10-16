"""Consume the one completed fixed-architecture scale result,publish,then stop."""
import fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TAG='aws_g_hier_composition_scale_20261016_v1'
RESULT=ROOT/f'experiments/results/hierarchy/{TAG}.json';STATE=ROOT/'.git/composition_scale_publication_20261016.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**fields):
 p=STATE.with_suffix('.writing');p.write_text(json.dumps(fields,indent=2)+'\n');p.replace(STATE)
def publish(r):
 assert r['status']=='completed' and r['stage']=='composition_scaling' and r['additional_updates']==2048
 refpath='experiments/results/hierarchy/aws_g_hier_grounded_decoder_only_20261016_v1.json';reference=json.loads((ROOT/refpath).read_text())['final']['materialized_dev_nll']
 f=r['final'];legacy=f['legacy']['primary_dev'];fresh=f['fresh_dev_legacy']['primary_dev'];broad=f['broad_train_readout']['primary_dev'];nll=f['materialized_dev_nll'];before=r['baseline']['materialized_dev_nll']
 improvement=nll<reference;strong=all(m['accuracy']>=.8 and m['nll']<.5 for m in [legacy,fresh])
 decision='The fixed architecture improves the saved whole-object development reference.' if improvement else 'The saved acquired actor retains the better original-protocol whole-object result.'
 if strong:decision+=' The predeclared strong composition milestone passes on original and fresh DEV.'
 paragraph=f"**Completed fixed-architecture composition scaling.** After2048additional paired updates over2048unique TRAIN scenes,original-protocol whole-object DEV NLL{before:.6f}→{nll:.6f},accuracy{100*legacy['accuracy']:.2f}%;saved acquired reference{reference:.6f}. Final fresh DEV NLL{fresh['nll']:.6f},accuracy{100*fresh['accuracy']:.2f}%. {decision} Broader TRAIN128 readout DEV NLL{broad['nll']:.6f} is a separately fitted endpoint. Same64TRAIN abstract arrangements,32new event/metric-layout tapes,two full pair epochs,unchanged model/Adam/loss/regularization,one development seed;TEST excluded,no ablation/control fitting. Source:`investment/composition_scale_evidence.json`."
 packet=dict(status='completed',battle='G-HIER',primary_dev_nll=nll,reference_dev_nll=reference,initial_dev_nll=before,primary_dev_accuracy=legacy['accuracy'],fresh_dev_nll=fresh['nll'],fresh_dev_accuracy=fresh['accuracy'],broader_train_readout_dev_nll=broad['nll'],shape_dev=f['shape_dev'],milestones=r['milestones'],settings=r['settings'],strong_milestone=strong,decision=decision,scope=r['scope'],work_training=r['work_training'],work_evaluation=r['work_evaluation'],input_sha256={str(RESULT.relative_to(ROOT)):sha(RESULT),refpath:sha(ROOT/refpath)})
 files=[]
 def write(name,s):(ROOT/name).write_text(s);files.append(name)
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert not (ROOT/'.git/MERGE_HEAD').exists()
  write('investment/composition_scale_evidence.json',json.dumps(packet,indent=2)+'\n')
  for name in ['experiments/HANDOFF.md','experiments/PRODUCT_ORDERS.md','experiments/hierarchy/CURRENT_FINDINGS.md']:
   s=(ROOT/name).read_text();marker='**Current composition scaling,16October:**'
   if marker in s:
    a=s.index(marker);b=s.index('\n\n',a);s=s[:a]+s[b+2:]
   write(name,paragraph+'\n\n'+s)
  for name in ['investment/PITCH.md','investment/INVESTMENT_CASE.md','investment/INVESTOR_PROOF_PLAN.md']:
   s=(ROOT/name).read_text();marker='**Composition scaling now tests data and optimization adequacy.**'
   if marker in s:
    a=s.index(marker);b=s.index('\n\n',a);s=s[:a]+paragraph+s[b:]
   else:
    a=s.index('\n\n');s=s[:a]+'\n\n'+paragraph+s[a:]
   write(name,s)
  name='REPORT.md';write(name,(ROOT/name).read_text()+'\n\n### Completed fixed-architecture composition data/training scaling\n\n'+paragraph+'\n\nMilestone128/512/2048 metrics,unique exposure,raw actor and calibrated-head scores,fresh final DEV,source/input/checkpoint bindings and all counted work are in the completed result. Data broadening preserves all original structural holdouts;new tapes do not expand the finite topology set.\n')
  name='report/I_SCIENCE.md';s=(ROOT/name).read_text();marker='**Composition scaling now tests data and optimization adequacy.**'
  if marker in s:
   a=s.index(marker);b=s.index('\n\n',a);s=s[:a]+paragraph+s[b:]
  else:
   a=s.index('\n## 5.');s=s[:a]+'\n\n'+paragraph+'\n'+s[a:]
  write(name,s)
  name='experiments/theory/COMPOSITION_DATA_AND_OPTIMIZATION_SCALING.md';write(name,(ROOT/name).read_text()+'\n\n## Completed endpoint\n\n'+paragraph+'\n')
  name='investment/brief/evidence_brief.tex';s=(ROOT/name).read_text();a=s.index('{\\small \\textbf{Identity-preserving composition.}');b=s.index('\\begin{center}',a)
  text=f"{{\\small \\textbf{{Fixed-architecture composition scaling.}} Ownership repair:25\\% support defect,8complete parts,19checks. After2048additional paired updates,original-protocol DEV NLL{nll:.4f} versus acquired{reference:.4f};fresh DEV{fresh['nll']:.4f}. Original/fresh accuracy{100*legacy['accuracy']:.1f}\\%/{100*fresh['accuracy']:.1f}\\%. Same64TRAIN arrangements,32new geometry/time tapes,unchanged model and regularization,no TEST or ablation arms. Source:\\texttt{{investment/composition\\_scale\\_evidence.json}}.}}\n\n"
  write(name,s[:a]+text+s[b:])
  scratch=ROOT/'.git/evidence-brief-preview/20261016CompositionScale/brief';scratch.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(ROOT/'investment/brief',scratch,dirs_exist_ok=True)
  env=os.environ.copy();local=ROOT/'experiments/.texmf_local';env.update(TEXMFHOME=str(local),TEXMFVAR=str(local/'var'),TEXMFCONFIG=str(local/'config'))
  with (scratch/'build.log').open('w') as out:subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','evidence_brief.tex'],cwd=scratch,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
  shutil.copy2(scratch/'evidence_brief.pdf',ROOT/'investment/sleeping_machines_evidence_brief.pdf');files.append('investment/sleeping_machines_evidence_brief.pdf')
  for script,args,label in [('report/make_pdf.py',['--current-record'],'report'),('scripts/build_investment_case.py',['--tag','20261016CompositionScale'],'memos'),('scripts/verify_investor_evidence.py',[],'verify'),('scripts/package_pitch_diligence.py',['--tag','20261016CompositionScale'],'bundle')]:
   with (ROOT/f'.git/composition_scale_{label}.log').open('w') as out:subprocess.run([str(ROOT/'.venv-docker/bin/python'),script,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
  files+=['investment/sleeping_machines_investment_case.pdf','investment/sleeping_machines_investor_pitch.pdf','investment/sleeping_machines_private_diligence.zip','report/sleeping_machines_status.pdf']+['report/'+n for n in ['I_SCIENCE.pdf','II_METHODS.md','II_METHODS.pdf','III_RECORD.md','III_RECORD.pdf']]
  subprocess.run(['git','add','--',*files],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish completed fixed-architecture composition scaling','--',*files],cwd=ROOT,check=True)
  pushed=subprocess.run(['git','push','origin','main'],cwd=ROOT).returncode==0
 status(stage='completed' if pushed else 'completed_local_pending_push',dev_nll=nll,fresh_dev_nll=fresh['nll'],strong_milestone=strong)
def main():
 if STATE.exists() and json.loads(STATE.read_text()).get('stage') in ['completed','completed_local_pending_push','needs_review']:return
 deadline=time.monotonic()+8*3600
 while time.monotonic()<deadline:
  if RESULT.exists() and (r:=json.loads(RESULT.read_text())).get('status')=='completed':publish(r);return
  partial=RESULT.with_suffix('.partial.json');milestones=[]
  if partial.exists():milestones=[dict(step=m['step'],nll=m['legacy']['primary_dev']['nll']) for m in json.loads(partial.read_text())['milestones']]
  status(stage='waiting_completed_fixed_architecture_fit',completed_milestones=milestones);time.sleep(30)
 raise RuntimeError('8hour wait expired;inspect exact checkpoint and run_safe status')
if __name__=='__main__':
 try:main()
 except Exception as error:status(stage='needs_review',error=repr(error));raise
