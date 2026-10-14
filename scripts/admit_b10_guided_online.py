"""Admit the strongest demonstrated native online guidance after actor reuse gates."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 plan=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z';marker=plan/'addenda/zzzzzzzzzzzzzzzzzzzzzzzzzzzzb10_guided_online_20261014_v1.json'
 if marker.exists():return
 required=['experiments/results/market/aws_b10_stable_native_s0_20261014_v1.json','experiments/results/market/aws_b10_online_native_s0_20261014_v1.json']
 while not all((ROOT/n).exists() and json.loads((ROOT/n).read_text()).get('status')=='completed' for n in required):time.sleep(30)
 names=['experiments/credit/aws_market_guided_online.py','experiments/credit/market_native_early_request.py','experiments/credit/market_native_remote_admission_selected.py','experiments/credit/native_market_credit.py','experiments/credit/market_streaming_clock_updates.py','experiments/market/paired_clock_credit.py','experiments/market/projected_clock_state.py','experiments/market/local_eligibility_state.py','experiments/market/episode_state.py','experiments/market/b10_stable_clock_head.py','experiments/market/b10_online_stream.py'];sources={n:sha(ROOT/n) for n in names};jobs=[];smoke='aws_b10_guided_online_smoke_20261014_v1'
 for tag,mode in [(smoke,'smoke'),('aws_b10_guided_online_s0_20261014_v1','fit')]:
  q=ROOT/f'experiments/queue/b10_online/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True);q.write_text(f'# B10: strongest demonstrated native online guidance on the same acquired actor/stream; source andBPTT parity required\n{tag} experiments/credit/aws_market_guided_online.py --tag {tag} --mode {mode}\n');requirements=[{'result':n} for n in required]
  if mode=='fit':requirements.append({'result':f'experiments/results/market/{smoke}.json','rss_margin':1.25})
  jobs.append(dict(name=tag,kind='b10_job',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=sources,result=f'experiments/results/market/{tag}.json',requires=requirements,timeout_s=1800 if mode=='smoke' else 10800,rss_kb=3000000,vms_kb=12000000))
 packet=dict(host='ip-172-31-47-132',battle='B10',purpose='Carry demonstrated learned native credit into online reference comparison; exact actor/acquisition/stream parity prevents hidden handicaps or extra warmup',parent_manifest_sha256=sha(plan/'manifest.json'),slots={'4':jobs});tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(marker);print('Admitted native guided online smoke and matched-stream fit',flush=True)
if __name__=='__main__':main()
