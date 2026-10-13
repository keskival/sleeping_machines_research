"""R1 next learned-residual fit after the reserved acquisition ladder."""
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+12*3600

def fit(reference,tag,mode,exact=False):
 r=queue.wait_result(ROOT/f'experiments/results/token_language/{reference}.json');sources=dict(r['source_sha256'])
 for name in ('experiments/residual_credit_token_language.py','sleeping_machines/predictable_route_credit.py'):sources[name]=queue.sha(ROOT/name)
 text=(ROOT/f'experiments/queue/r1/{reference}/{reference}.txt').read_text().splitlines()[1]
 args=text.split('.py ',1)[1].replace(reference,tag)+f' --credit-predictor {mode}'
 result=f'experiments/results/token_language/{tag}.json';admit(tag,'experiments/residual_credit_token_language.py',args,result,sources,[],True);d=queue.wait_result(ROOT/result);sources.update(d['source_sha256'])
 driver='experiments/analyze_residual_credit.py';sources[driver]=queue.sha(ROOT/driver);check=tag+'_gate';result=f'experiments/results/diagnostics/{check}.json'
 admit(check,driver,f'--tag {check} --reference {reference} --candidate {tag}'+(' --exact' if exact else ''),result,sources,[],False)
 return queue.wait_result(ROOT/result)['gate']=='PASS'

def main():
 two=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_bernoulli_teacher_2m_20261013_s6_v2_quality_priority_v2.json')
 if two['accepted']:queue.wait_result(ROOT/'experiments/results/diagnostics/aws_bernoulli_teacher_8m_20261013_s6_v2_quality_priority_v2.json')
 math=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_predictable_route_credit_contract_20261013_v1.json');assert math['gate']=='PASS'
 if not fit('aws_bernoulli_teacher_exact_20261013_s6_v1','aws_residual_credit_off_20261013_s6_v1','off',True):print('STOP: off-mode parity failed',flush=True);return
 if not fit('aws_bernoulli_teacher_128k_20261013_s6_v1','aws_residual_credit_128k_20261013_s6_v1','linear'):print('STOP: predictor needs error analysis; retained recipe unchanged',flush=True);return
 fit('aws_bernoulli_teacher_512k_20261013_s6_v1','aws_residual_credit_512k_20261013_s6_v1','linear')
 print('Bounded residual-credit integration complete',flush=True)
if __name__=='__main__':main()
