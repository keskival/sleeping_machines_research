"""R1 user-directed256-token credit acquisition after guarded gates."""
import math,time
import continue_local_clip_language as queue
ROOT=queue.ROOT;queue.queue.DEADLINE=time.monotonic()+30*86400

def main():
 gate=queue.queue.wait_result(ROOT/'experiments/results/credit/aws_unrestricted_language_contract_20261014_v1.json');assert gate['gate']=='PASS'
 smoke=queue.queue.wait_result(ROOT/'experiments/results/token_language/aws_unrestricted_d8_credit256_smoke_20261014_s6_v1.json')
 assert smoke['max_rss_kb']<2500000
 for layer in smoke['curve'][-1]['receiver_gradient_norms']['banks.input']:
  assert sum(v for head in layer for v in head)>0
 tag='aws_unrestricted_d8_credit256_8m_20261014_s6_v1'
 template=(ROOT/'experiments/queue/r1/aws_unrestricted_d8_credit256_smoke_20261014_s6_v1/aws_unrestricted_d8_credit256_smoke_20261014_s6_v1.txt').read_text().splitlines()[1].split('.py ',1)[1]
 args=template.replace('aws_unrestricted_d8_credit256_smoke_20261014_s6_v1',tag).replace('--steps 16','--steps 4096').replace('--eval-every 16','--eval-every 256')
 timeout=math.ceil(2*smoke['train_wall_s_total']/smoke['presentations_total']*8388608+3600)
 # Resource estimates determine the envelope; no quality gate or arbitrary context cap.
 queue.DRIVER='experiments/unrestricted_receiver_token_language.py'
 result=queue.admit(tag,args,smoke['source_sha256'],max(timeout,5400))
 print('completed '+tag+' actualtargets '+str(result['presentations_total'])+' finalDEV '+str(result['curve'][-1]['dev_nll']),flush=True)
if __name__=='__main__':main()
