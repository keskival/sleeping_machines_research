"""Three causal repairs, one shared acquired reference; six-slot AWS only."""
import json,math,time
from concurrent.futures import ThreadPoolExecutor
import continue_calibrated_clip_pair as queue
ROOT=queue.ROOT;queue.queue.DEADLINE=time.monotonic()+30*86400

def arm(variant,slot):
 r=queue.queue.wait_result(ROOT/'experiments/results/token_language/aws_integrated_receiver_d8_8m_20261014_s6_v1.json');sources=dict(r['source_sha256'])
 for name in ['experiments/scaling_repair_continuation.py','sleeping_machines/contained_nonlinear_readout.py','sleeping_machines/route_credit_horizon.py','sleeping_machines/learning/calibrated_receiver_adamw.py','experiments/credit/receiver_allocated_future_world.py','sleeping_machines/learning/clone_receiver_learner.py','experiments/credit/native_token_teacher_world.py','experiments/credit/native_token_future_world.py']:sources[name]=queue.queue.sha(ROOT/name)
 prefix=f'aws_scaling_repair_{variant}_20261014_s6_v1';smoke=prefix+'_smoke';driver='experiments/scaling_repair_continuation.py';s=queue.admit(smoke,driver,f'--tag {smoke} --variant {variant} --policy global --updates 64 --eval-every 64',f'experiments/results/token_language/{smoke}.json',sources,slot,1800)
 assert s['max_rss_kb']<2500000 and s['local_cap_gate']
 if variant=='decoder':assert s['readout_containment']['normalized_loss_parity'] and s['readout_containment']['feature_cotangent_parity']
 timeout=math.ceil(2*s['added_training_s']/s['added_targets']*2097152+3600);tag=prefix+'_2m'
 result=queue.admit(tag,driver,f'--tag {tag} --variant {variant} --policy global --updates 4096 --eval-every 1024',f'experiments/results/token_language/{tag}.json',sources,slot,max(timeout,3600),[dict(result=f'experiments/results/token_language/{smoke}.json')])
 sources=dict(result['source_sha256'])
 for name in ['experiments/score_scaling_repair.py','experiments/public_token_scoring.py']:sources[name]=queue.queue.sha(ROOT/name)
 q=tag+'_quality';d=queue.admit(q,'experiments/score_scaling_repair.py',f'--tag {q} --run {tag}',f'experiments/results/diagnostics/{q}.json',sources,slot,3600)
 reference=queue.queue.wait_result(ROOT/'experiments/results/diagnostics/aws_calibrated_clip_pair_global_20261014_s6_v1_2m_quality.json');g=reference['rows'][-1]
 for k in ('targets','target_start','target_end_inclusive','window','lanes','eos_state_reset'):assert g['scored'][k]==d['scored'][k]
 assert g['acquisition']==result['presentations_total']
 print(json.dumps(dict(variant=variant,gain_vs_matched_global=g['scored']['nll']-d['scored']['nll'],added_training_s=result['added_training_s'],smoke_training_s=s['added_training_s'])),flush=True)

def main():
 with ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(arm,v,slot) for v,slot in [('horizon',1),('long_factual',3),('decoder',6)]]
  for f in futures:f.result()
 print('Three causal comparisons complete; no automatic larger acquisition',flush=True)
if __name__=='__main__':main()
