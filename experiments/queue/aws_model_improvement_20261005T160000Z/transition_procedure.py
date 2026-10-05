import sys,os,signal,time,json,fcntl,hashlib,shutil
from pathlib import Path
sys.path.insert(0,'scripts');import publish_aws_language_progress as P
old='experiments/queue/aws_product_continue_20261005T073000Z/manifest.json';new=Path('experiments/queue/aws_model_improvement_20261005T160000Z')
assert (new/'manifest.json').exists()
def processes():
 d={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   f=(p/'stat').read_text().rsplit(')',1)[1].split();d[int(p.name)]=(int(f[1]),f[19],f[0])
  except (OSError,IndexError):pass
 return d
with open('/tmp/aws-language-publication.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);pid=P.coordinator(old);os.kill(pid,signal.SIGSTOP)
 try:
  assert not P.git_children(pid)
  tree=processes();targets={pid};changed=True
  while changed:
   additions={p for p,(parent,_,_) in tree.items() if parent in targets};changed=bool(additions-targets);targets|=additions
  publishers=[]
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:a=(p/'cmdline').read_bytes().split(b'\0')
   except OSError:continue
   if b'scripts/publish_aws_product_progress.py' in a and old.encode() in a:publishers.append(int(p.name))
  for p in publishers:os.kill(p,signal.SIGTERM)
  for p in targets:
   try:os.kill(p,signal.SIGTERM)
   except ProcessLookupError:pass
  os.kill(pid,signal.SIGCONT)
  for _ in range(30):
   current=processes();alive=[p for p in targets if p in current and current[p][1]==tree[p][1] and current[p][2]!='Z']
   if not alive:break
   time.sleep(1)
  else:raise RuntimeError('Old guarded descendants remain '+str(alive))
  import torch;torch.set_num_threads(1)
  job=json.loads((new/'manifest.json').read_text())['slots']['1'][0]
  cp=Path(job['result']).parent/'checkpoints'/(job['tag']+'.pt');archive=new/'native_resume.pt';shutil.copyfile(cp,archive);saved=torch.load(archive,weights_only=False)
  record=dict(status='checkpointed_transition',old_manifest=old,new_manifest=str(new/'manifest.json'),stopped_processes=sorted(targets),seen=saved['seen'],window=saved['window'],wall_s=saved['wall_s'],checkpoint=str(archive),checkpoint_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),discarded_presentations_upper_bound=4095999,scope='Exact saved model/Adam/schedule/RNG/cursor resumed. Uncheckpointed prior computation unknown, bounded by 500 windows. Race-only primate excludes unreachable expected-reception imports; executed sources unchanged.')
  (new/'transition.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
 except:
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
  raise
