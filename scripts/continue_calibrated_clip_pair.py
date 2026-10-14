"""One paired R1 clipping repair in six-slot AWS gym; gates then2M continuation."""
import json,math,time
import continue_window_replay as queue
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+30*86400
DRIVER='experiments/calibrated_clip_continuation.py'

def admit(tag,script,args,result,sources,slot,timeout,requires=()):
 p=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';text='# R1: matched continuation localclip repair; decision quality and completework versus same acquired global learner\n'+tag+' '+script+' '+args+'\n'
 if p.exists():assert p.read_text()==text;return queue.wait_result(ROOT/result)
 p.parent.mkdir(exist_ok=False);p.write_text(text)
 job=dict(name=tag,queue=str(p.relative_to(ROOT)),queue_sha256=queue.sha(p),kind='r1_job',source_sha256=sources,result=result,requires=list(requires),timeout_s=timeout,rss_kb=3000000,vms_kb=12000000)
 packet=dict(host='ip-172-31-47-132',battle='R1',purpose='Matched calibratedlocal/global continuation, one fullmodel repair pair, no externalarchitecture or publicselection',parent_manifest_sha256=queue.sha(queue.PLAN/'manifest.json'),slots={str(slot):[job]})
 path=queue.PLAN/'addenda'/('zzzzzzzzzzzzzzzzzzzzzzzzzzzr1_'+str(time.time_ns())+'_s'+str(slot)+'.json');path.write_text(json.dumps(packet,indent=2)+'\n');print('admitted '+tag,flush=True)
 return queue.wait_result(ROOT/result)

def arm(policy,slot):
 ref='aws_integrated_receiver_d8_8m_20261014_s6_v1';r=queue.wait_result(ROOT/f'experiments/results/token_language/{ref}.json');sources=dict(r['source_sha256'])
 names=[DRIVER,'sleeping_machines/learning/calibrated_receiver_adamw.py','sleeping_machines/learning/clone_receiver_learner.py','experiments/credit/receiver_allocated_future_world.py','experiments/credit/native_token_teacher_world.py','experiments/credit/native_token_future_world.py']
 for name in names:sources[name]=queue.sha(ROOT/name)
 prefix='aws_calibrated_clip_pair_'+policy+'_20261014_s6_v1';smoke=prefix+'_smoke';s=admit(smoke,DRIVER,f'--tag {smoke} --policy {policy} --updates 64 --eval-every 64',f'experiments/results/token_language/{smoke}.json',sources,slot,1800)
 assert s['local_cap_gate'] and s['max_rss_kb']<2500000
 if policy=='global':assert s['global_first_update_parity']
 timeout=math.ceil(2*s['added_training_s']/s['added_targets']*2097152+3600)
 tag=prefix+'_2m';r=admit(tag,DRIVER,f'--tag {tag} --policy {policy} --updates 4096 --eval-every 1024',f'experiments/results/token_language/{tag}.json',sources,slot,max(timeout,3600),[dict(result=f'experiments/results/token_language/{smoke}.json')])
 sources=dict(r['source_sha256'])
 for name in ['experiments/check_deep_language_quality.py','experiments/public_token_scoring.py','experiments/bernoulli_teacher_token_language.py']:sources[name]=queue.sha(ROOT/name)
 q=tag+'_quality';quality=admit(q,'experiments/check_deep_language_quality.py',f'--tag {q} --run {tag}',f'experiments/results/diagnostics/{q}.json',sources,slot,3600)
 print(json.dumps(dict(policy=policy,quality=quality['rows'][-1]['scored']['nll'],added_training_s=r['added_training_s'],calibration_wall_s=r['calibration_wall_s'],smoke_training_s=s['added_training_s'])),flush=True)

def main():
 # Two independent arms begin from the identical completed cut; smoke is not quality selection.
 from concurrent.futures import ThreadPoolExecutor
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(arm,policy,slot) for policy,slot in [('global',4),('calibrated',5)]]
  for f in futures:f.result()
 print('Pair completed; no automatic horizon/scale continuation',flush=True)
if __name__=='__main__':main()
