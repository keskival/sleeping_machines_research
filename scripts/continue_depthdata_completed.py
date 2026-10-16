"""Publish the two admitted AWS depth-data endpoints when both complete.

No trainer/evaluator is launched. This consumes completed source-bound JSON
only,updates the current documents/PDFs once,and stops.
"""
import fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.git/depthdata_completed_publication_20261016.json'
RUNS={d:ROOT/f'experiments/results/token_language/aws_r1_depthdata_L{d}_16m_s0_20261016_v1.json' for d in [4,16]}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**values):
 tmp=STATE.with_suffix('.writing');tmp.write_text(json.dumps(values,indent=2)+'\n');tmp.replace(STATE)
def publish(results):
 for d,r in results.items():
  assert r['status']=='completed' and r['args']['layers']==d and r['args']['train_tokens']==16777216
 for k in ['seed','train_tokens','lane','d','modes','heads','dk','dropout','lr','wd','batch']:
  assert results[4]['args'][k]==results[16]['args'][k],k
 assert results[4]['steps']==results[16]['steps']
 rows=[dict(layers=d,nll=r['flagship_protocol']['nll'],last_quarter_prequential=r['prequential_last_quarter'],parameters=r['parameters'],presentations=r['presentations'],max_rss_kb=r['max_rss_kb'],train_wall_s=r['train_wall_s']) for d,r in results.items()]
 scope='Dense hybrid diagnostic only. One seed,one pass over16M fresh FineWeb GPT2 tokens,equal update count,different parameters/work. Final8192target/1024window flagship DEV protocol,no public TEST or native-model verdict. Complete learning arithmetic not audited.'
 packet=dict(status='completed',battle='R1',rows=rows,scope=scope,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in RUNS.values()})
 a,b=rows;delta=a['nll']-b['nll'];decision=('L16 improves over L4 in this pair.' if delta>0 else 'L4 leads L16 in this pair;diagnose the extra depth before extending it.')
 paragraph=f"**Completed 16M hybrid diagnostic.** L4 flagship DEV NLL{a['nll']:.4f};L16 {b['nll']:.4f}. {decision} {scope} Source:`investment/depthdata_16m_evidence.json`."
 files=[]
 def write(name,text):
  (ROOT/name).write_text(text);files.append(name)
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert not (ROOT/'.git/MERGE_HEAD').exists(),'finish existing merge before publication'
  write('investment/depthdata_16m_evidence.json',json.dumps(packet,indent=2)+'\n')
  for p in ['experiments/HANDOFF.md','REPORT.md','report/I_SCIENCE.md','investment/PITCH.md','investment/INVESTMENT_CASE.md','investment/INVESTOR_PROOF_PLAN.md']:
   s=(ROOT/p).read_text();marker='**Completed 16M hybrid diagnostic.**'
   if marker in s:
    at=s.index(marker);end=s.index('\n\n',at);s=s[:at]+paragraph+s[end:]
   elif p.startswith('investment/'):
    at=s.index('**Native continuation.**');end=s.index('\n\n',at);s=s[:end]+'\n\n'+paragraph+s[end:]
   else:s+='\n\n'+paragraph+'\n'
   write(p,s)
  p='investment/brief/evidence_brief.tex';s=(ROOT/p).read_text();anchor='\\textbf{Memories compose, then emit.}'
  assert anchor in s
  line=f"\\textbf{{Completed 16M hybrid diagnostic.}} Four layers: \\textbf{{{a['nll']:.4f}}}; sixteen: \\textbf{{{b['nll']:.4f}}} flagship DEV NLL. One pass, one seed, equal updates and different work. This is the native model's target,not a native win. Source: \\path{{investment/depthdata_16m_evidence.json}}.\n\n"
  marker='\\textbf{Completed 16M hybrid diagnostic.}'
  if marker in s:
   at=s.index(marker);end=s.index('\n\n',at);s=s[:at]+line.rstrip()+s[end:]
  else:s=s.replace(anchor,line+anchor,1)
  write(p,s)
  p='investment/claims_registry.json';reg=json.loads((ROOT/p).read_text())
  ids={c['id'] for c in reg['claims']}
  for i,row in enumerate(rows):
   ident=f"hybrid16M_pair_L{row['layers']}_flagship"
   if ident not in ids:reg['claims'].append(dict(id=ident,display=f"{row['nll']:.4f}",decimals=4,value=dict(file='investment/depthdata_16m_evidence.json',path=['rows',i,'nll']),documents=['investment/brief/evidence_brief.tex']))
  write(p,json.dumps(reg,indent=2)+'\n')
  scratch=ROOT/'.git/evidence-brief-preview/20261016DepthData/brief';scratch.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(ROOT/'investment/brief',scratch,dirs_exist_ok=True)
  env=os.environ.copy();local=ROOT/'experiments/.texmf_local';env.update(TEXMFHOME=str(local),TEXMFVAR=str(local/'var'),TEXMFCONFIG=str(local/'config'))
  with (scratch/'build.log').open('w') as log:subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','evidence_brief.tex'],cwd=scratch,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  shutil.copy2(scratch/'evidence_brief.pdf',ROOT/'investment/sleeping_machines_evidence_brief.pdf');files.append('investment/sleeping_machines_evidence_brief.pdf')
  with (ROOT/'.git/depthdata_report_build.log').open('w') as log:subprocess.run([str(ROOT/'.venv-docker/bin/python'),'report/make_pdf.py','--current-record'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  files+=['report/sleeping_machines_status.pdf']+['report/'+n for n in ['I_SCIENCE.pdf','II_METHODS.md','II_METHODS.pdf','III_RECORD.md','III_RECORD.pdf']]
  for script,args,label in [('scripts/build_investment_case.py',['--tag','20261016DepthData'],'memos'),('scripts/package_pitch_diligence.py',['--tag','20261016DepthData'],'bundle')]:
   with (ROOT/f'.git/depthdata_{label}.log').open('w') as log:subprocess.run([str(ROOT/'.venv-docker/bin/python'),script,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  files+=['investment/sleeping_machines_investment_case.pdf','investment/sleeping_machines_investor_pitch.pdf','investment/sleeping_machines_private_diligence.zip']
  subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/verify_investor_evidence.py'],cwd=ROOT,check=True)
  subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
  subprocess.run(['git','commit','--only','-m','Publish completed16M native depth-data evidence','--',*files],cwd=ROOT,check=True)
  pushed=subprocess.run(['git','push','origin','main'],cwd=ROOT).returncode==0
 status(stage='completed' if pushed else 'completed_local_pending_push',rows=rows,decision=decision)
def main():
 if STATE.exists() and json.loads(STATE.read_text()).get('stage') in ['completed','completed_local_pending_push','needs_review']:return
 deadline=time.monotonic()+36*3600
 while time.monotonic()<deadline:
  results={d:json.loads(p.read_text()) for d,p in RUNS.items() if p.exists()}
  if len(results)==2 and all(r.get('status')=='completed' for r in results.values()):publish(results);return
  status(stage='waiting_completed_endpoints',available=list(results));time.sleep(30)
 raise RuntimeError('36hour publication wait expired;inspect run_safe/checkpoints')
if __name__=='__main__':
 try:main()
 except Exception as error:status(stage='needs_review',error=repr(error));raise
