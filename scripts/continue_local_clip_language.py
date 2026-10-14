"""R1 one integrated D8 candidate, source/resource gate then reserved8M."""
import json,math,time
import continue_window_replay as queue
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+14*86400
DRIVER='experiments/local_clip_receiver_token_language.py'
def admit(tag,args,sources,timeout):
 p=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt'
 text='# R1: integrated native language; decision quality/full learning work versus saved full-teaching and deep references\n'+tag+' '+DRIVER+' '+args+'\n'
 if p.exists():
  assert p.read_text()==text
  return queue.wait_result(ROOT/f'experiments/results/token_language/{tag}.json')
 p.parent.mkdir(exist_ok=False);p.write_text(text)
 job=dict(name=tag,queue=str(p.relative_to(ROOT)),queue_sha256=queue.sha(p),kind='r1_job',source_sha256=sources,result=f'experiments/results/token_language/{tag}.json',requires=[],timeout_s=timeout,rss_kb=3000000,vms_kb=12000000)
 packet=dict(host='ip-172-31-47-132',battle='R1',purpose='User-directed per-bank clipping replacement; compare same acquired data and complete work',parent_manifest_sha256=queue.sha(queue.PLAN/'manifest.json'),slots={'1':[job]})
 p=queue.PLAN/'addenda'/('zzzzzzzzzzzzzzzzzzzzzzzzzzzr1_'+str(time.time_ns())+'_s1.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(p)
 print('admitted '+tag,flush=True);return queue.wait_result(ROOT/job['result'])
def main():
 gate=queue.wait_result(ROOT/'experiments/results/credit/aws_local_clip_independence_contract_20261014_v1.json');assert gate['gate']=='PASS'
 ref='aws_native_deep_d8_8m_20261013_s6_v1';r=queue.wait_result(ROOT/f'experiments/results/token_language/{ref}.json')
 sources=dict(r['source_sha256']);sources.pop('experiments/bernoulli_teacher_token_language.py')
 for name in [DRIVER,'sleeping_machines/depth_credit_allocator.py','sleeping_machines/learning/receiver_bank_adamw.py','sleeping_machines/learning/receiver_credit_banks.py','sleeping_machines/learning/local_credit_banks.py']:sources[name]=queue.sha(ROOT/name)
 template=(ROOT/f'experiments/queue/r1/{ref}/{ref}.txt').read_text().splitlines()[1].split('.py ',1)[1]
 smoke='aws_local_clip_receiver_d8_smoke_20261014_s6_v1'
 args=template.replace(ref,smoke).replace('--steps 16384','--steps 64').replace('--eval-every 1024','--eval-every 64')
 s=admit(smoke,args,sources,5400)
 assert s['max_rss_kb']<2500000
 for layer in s['curve'][-1]['receiver_gradient_norms']['banks.input']:
  assert sum(v for head in layer for v in head)>0
 seconds=s['train_wall_s_total']/s['presentations_total'];timeout=math.ceil(2*seconds*8388352+3600)
 if timeout>7*86400:raise ValueError('Measured fit exceeds7day envelope; optimize execution before scaling')
 tag='aws_local_clip_receiver_d8_8m_20261014_s6_v1';r=admit(tag,template.replace(ref,tag),sources,max(timeout,5400))
 print('completed integrated8M finalDEV '+str(r['curve'][-1]['dev_nll'])+'; trainingseconds '+str(r['train_wall_s_total']),flush=True)
 print('Next source-bound paired quality/work and production-horizon audit; publicvalidationsealed',flush=True)
if __name__=='__main__':main()
