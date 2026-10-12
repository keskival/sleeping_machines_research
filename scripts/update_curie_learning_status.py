"""Publish a short current Curie verdict, retaining detailed source-bound appendices."""
import fcntl,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  text='**Forward-taught credit and native learning (12 October):** Generated-credit full-cohort depth8/16/32 trails native AdamW. The fixed three-seed hidden-bias scalar intervention effect changes sign and is retired. The connected critic now trains its credit vector through the actual clipped-Adam update and complete next-window factual loss; both audit branches pass finite differences. The earlier detached-teacher v2 fit is a protocol error, preserved as data and excluded from method claims. '
  paths=[('curie_market_prospective_bias_20261012_v1_fit','Prospective mean-credit critic'),('curie_market_functional_future_20261012_v1_fit','Operator-informed functional critic'),('curie_market_direct_backtrack_20261012_v1_fit','Revealed-future descent constrained critic'),('curie_market_direct_future_20261012_v3_fit','Corrected direct future critic')]
  for tag,label in paths:
   p=ROOT/f'experiments/results/credit/{tag}.json'
   if p.exists():
    r=json.loads(p.read_text());assert r['status']=='completed';v={x['arm']:x for x in r['rows']}
    text+=label+': '+', '.join(f"{x['arm']} NLL {x['nll']:.6f}, reuse {x['wall_s']:.3f}s" for x in r['rows'])+f"; shared acquisition {r['acquisition']['wall_s']:.3f}s. "
    losses=[a for a in ['frozen','zero','bptt'] if v['online']['nll']>=v[a]['nll']]
    text+='Online trails '+', '.join(losses)+'. ' if losses else 'Online improves quality versus all three controls in this development run. '
    break
  completed=[]
  for seed in [187,188,189]:
   tag='curie_market_functional_future_20261012_v1_fit' if seed==187 else f'curie_market_functional_future_20261012_v1_s{seed}_fit'
   p=ROOT/f'experiments/results/credit/{tag}.json'
   if p.exists():
    r=json.loads(p.read_text());assert r['status']=='completed';v={x['arm']:x for x in r['rows']};completed.append(v['bptt']['nll']-v['online']['nll'])
  if len(completed)==3:text+=f"Fixed confirmation wins {sum(x>0 for x in completed)}/3 versus shared-start native BPTT, mean gain {sum(completed)/3:+.6f}; the first-seed quality win is not a confirmed advantage. Short-fragment tuning stops. "
  p=ROOT/'experiments/results/credit/curie_market_functional_large_20261012_v1_fit.json'
  if p.exists():
   r=json.loads(p.read_text());assert r['status']=='completed';v={x['arm']:x for x in r['rows']};b=v['bptt_from_initialization'];text+=f"Longer-data whole-learning result: online {r['online_whole_nll']:.6f} nats/event/{r['online_whole_cpu_s']:.3f}s, native BPTT from original initialization {b['nll']:.6f}/{b['wall_s']:.3f}s, on65,536chronological TRAIN targets. "
  text+='Chronological Aug1 TRAIN only; each source records acquisition/continuation sizes and seeds; all producer copies, audits, teacher/candidate replays and optimizer work are charged. Dense v5 temporal/mark-memory boundary integration; no sealed TEST, optional-route or hardware claim. Next development is decided by future quality against zero/native BPTT and complete cost, not cotangent MSE or theory alone.'
  p=ROOT/'report/I_SCIENCE.md';s=p.read_text();marker='**Forward-taught credit and native learning (12 October):**';start=s.index(marker);end=s.index('\n',start);p.write_text(s[:start]+text+s[end:])
if __name__=='__main__':main()
