"""R1 native temporal depth8/16 and acquisition8M/32M, host-guarded."""
import json,math,time
import continue_window_replay as queue
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+7*86400
DRIVER='experiments/bernoulli_teacher_token_language.py'


def admit(tag,args,sources,timeout):
 path=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';path.parent.mkdir(exist_ok=False)
 path.write_text('# R1: temporal depth and acquisition scaling; decision useful deep prediction versus complete learning work\n'+tag+' '+DRIVER+' '+args+'\n')
 result=f'experiments/results/token_language/{tag}.json'
 job=dict(name=tag,queue=str(path.relative_to(ROOT)),queue_sha256=queue.sha(path),kind='r1_job',source_sha256=sources,result=result,requires=[],timeout_s=timeout,rss_kb=3000000,vms_kb=12000000)
 packet=dict(host='ip-172-31-47-132',battle='R1',purpose='User-directed native temporal depth8/16 and acquisition8M/32M; no small-quality veto',parent_manifest_sha256=queue.sha(queue.PLAN/'manifest.json'),slots={'2':[job]})
 path=queue.PLAN/'addenda'/('zzzzzzzzzzzzzzzzzzzzzzzzzzzr1_'+str(time.time_ns())+'_s2.json');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(path)
 print('admitted '+tag,flush=True);return queue.wait_result(ROOT/result)


def finite(value):
 if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
 if isinstance(value,dict):return all(finite(v) for v in value.values())
 return value is None or not isinstance(value,(int,float)) or math.isfinite(value)


def main():
 reference='aws_bernoulli_teacher_8m_20261013_s6_v2'
 base=queue.wait_result(ROOT/f'experiments/results/token_language/{reference}.json');sources=dict(base['source_sha256'])
 for name,digest in sources.items():assert queue.sha(ROOT/name)==digest,name
 template=(ROOT/f'experiments/queue/r1/{reference}/{reference}.txt').read_text().splitlines()[1].split('.py ',1)[1]
 smokes={}
 for depth in (8,16):
  tag=f'aws_native_deep_d{depth}_smoke_20261013_s6_v1'
  args=template.replace(reference,tag).replace('--depth 2',f'--depth {depth}').replace('--steps 16384','--steps 64').replace('--eval-every 256','--eval-every 64')
  r=admit(tag,args,sources,5400);assert finite(r['curve'])
  norms=r['curve'][-1]['receiver_gradient_norms']['banks.input'];assert len(norms)==depth
  for layer in norms:assert sum(v for head in layer for v in head)>0,'Missing layer input gradient'
  assert r['max_rss_kb']<2500000
  smokes[depth]=r['train_wall_s_total']/r['presentations_total']
  print('PASS depth'+str(depth)+' finite fit/every-layer gradient; seconds/target '+str(smokes[depth]),flush=True)
 for depth,size,steps in ((8,'8m',16384),(16,'8m',16384),(8,'32m',65536)):
  tag=f'aws_native_deep_d{depth}_{size}_20261013_s6_v1'
  args=template.replace(reference,tag).replace('--depth 2',f'--depth {depth}').replace('--steps 16384',f'--steps {steps}').replace('--eval-every 256','--eval-every 1024')
  # Measured smoke throughput plus evaluation/headroom; long runs retain RSS watchdog.
  timeout=math.ceil(2*smokes[depth]*steps*512+3600)
  if timeout>172800:raise RuntimeError('Measured fit exceeds48h envelope; optimize deep execution without shrinking depth')
  r=admit(tag,args,sources,max(timeout,5400));assert finite(r['curve'])
  print('Completed '+tag+'; finalDEV '+str(r['curve'][-1]['dev_nll'])+'; training_s '+str(r['train_wall_s_total']),flush=True)
 print('Depth8/16 and32M acquisition complete; compare quality/work and long-history DEV, publicvalidation sealed',flush=True)
if __name__=='__main__':main()
