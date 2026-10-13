"""R1 p1 numerical witness -> one p.25 fit -> quality/work gate -> larger fit."""
import json,time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+4*3600

def run(reference,tag,prob,exact=False):
 old=queue.wait_result(ROOT/f'experiments/results/token_language/{reference}.json')
 text=(ROOT/f'experiments/queue/r1/{reference}/{reference}.txt').read_text().splitlines()[1]
 driver='experiments/bernoulli_teacher_token_language.py'
 args=text.split('.py ',1)[1].replace(reference,tag)+f' --teacher-probability {prob}'
 sources=dict(old['source_sha256']);sources[driver]=queue.sha(ROOT/driver)
 for source in ('sleeping_machines/ordered_token_lookup.py','sleeping_machines/ordered_token_episodes.py'):sources[source]=queue.sha(ROOT/source)
 result=f'experiments/results/token_language/{tag}.json';admit(tag,driver,args,result,sources,[],True);d=queue.wait_result(ROOT/result)
 sources.update(d['source_sha256']);check=tag+'_gate';driver='experiments/analyze_bernoulli_teacher.py';sources[driver]=queue.sha(ROOT/driver)
 result=f'experiments/results/diagnostics/{check}.json'
 admit(check,driver,f'--tag {check} --reference {reference} --candidate {tag}'+(' --exact' if exact else ''),result,sources,[],False)
 proof=queue.wait_result(ROOT/result);print(json.dumps(proof),flush=True)
 return proof['gate']=='PASS'

def main():
 if not run('aws_native_work_reference_20261013_v1','aws_bernoulli_teacher_exact_20261013_s6_v1',1,True):return
 if not run('aws_integrated_ordered_lookup_128k_20261013_s6_v2','aws_bernoulli_teacher_128k_20261013_s6_v1',.25):return
 run('aws_integrated_ordered_lookup_512k_20261013_s6_v3','aws_bernoulli_teacher_512k_20261013_s6_v1',.25)
 print('Bounded teacher-admission development complete; inspect gates before further scaling',flush=True)
if __name__=='__main__':main()
