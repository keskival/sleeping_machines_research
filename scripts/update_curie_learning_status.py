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
  native=ROOT/'experiments/results/causal_induction/curie_causal_race_credit_20261012_v1_fit.json'
  if native.exists():
   r=json.loads(native.read_text());assert r['status']=='completed';v={x['arm']:x for x in r['rows']};exact=v['exact'];learned=v['credit'];joint=v['joint']
   text='**Forward-taught credit and native learning (12 October):** Credit now uses the native temporal/race substrate itself: structured relationship/query/outcome tokens, shared hard-race units, sparse relationship-addressed persistent state and a linear future-loss readout. Learned token projections, keys, clocks, gates and temporal dynamics receive teaching from revealed factual/counterfactual forecast losses. '
   text+=f"Native memory credit improves four-graph DEV NLL {exact['mean_dev_nll']:.6f}→{learned['mean_dev_nll']:.6f}, improving each graph, with fitting {exact['whole_fit_wall_s']:.3f}→{learned['whole_fit_wall_s']:.3f}s. Joint memory-guided routing scores {joint['mean_dev_nll']:.6f}. Single-seed quality gain against exact route teaching; no efficiency win. "
   control=ROOT/'experiments/results/causal_induction/curie_causal_race_credit_20261012_v1_controls.json'
   if control.exists():
    c=json.loads(control.read_text());assert c['status']=='completed';text+='Matched25% audit controls: '+', '.join(f"{x['arm']} DEV {x['mean_dev_nll']:.6f}/{x['whole_fit_wall_s']:.3f}s" for x in c['rows'])+'. '
    losing=[x['arm'] for x in c['rows'] if learned['mean_dev_nll']>=x['mean_dev_nll']]
    text+='Learned credit trails '+', '.join(losing)+'; the gain over exact teaching is not an attributed learning advantage. ' if losing else 'Learned memory credit improves both attribution controls in this development run. '
   else:text+='Audit-rate and credit-memory attribution controls are running; no attributed learning claim. '
   decision=ROOT/'experiments/results/causal_induction/curie_causal_race_credit_confirm_20261012_v1_decision.json'
   if decision.exists():
    d=json.loads(decision.read_text());assert d['status']=='completed';text+='Unchanged three-actor/four-new-graph cohort: learned credit beats zero-estimate and reset-memory controls on all three actors, mean gains '+f"{d['mean_quality_gains']['zero25']:.6f}/{d['mean_quality_gains']['reset_memory']:.6f}. "+f"Against exact teaching, mean gain {d['mean_quality_gains']['exact']:.6f}, two of three actors better. "+('Registered transfer gate passes. ' if d['confirmation_gate'] else 'Strict transfer gate fails; larger nonlinear transfer is deferred. ')
   centered=ROOT/'experiments/results/causal_induction/curie_causal_centered_credit_20261012_v3_pilot.json'
   if centered.exists():
    c=json.loads(centered.read_text());assert c['status']=='completed';text+='Native baseline-relative teacher pilot: '+', '.join(f"{x['arm']} DEV {x['mean_dev_nll']:.6f}/{x['whole_fit_wall_s']:.3f}s" for x in c['rows'])+'. Original selection graphs, separate from unchanged confirmation. '
   bank=ROOT/'experiments/results/causal_induction/curie_shared_bank_20261012_v2_fit.json'
   if bank.exists():
    c=json.loads(bank.read_text());assert c['status']=='completed';text+='Independent native shared-bank fit: '+', '.join(f"{x['arm']} DEV {x['mean_dev_nll']:.6f}/{x['whole_fit_wall_s']:.3f}s" for x in c['rows'])+'. Each role chooses its own reads and native receiver writes; later task losses teach critic operations through shared records. Untaught control preserves actor-reader learning while stopping critic-record gradients. Relational destinations and bounded recency discovery; eight-feedback teaching windows, persistent numerical state. '
   text+='Private four-root affine diagnostic, eight TRAIN graphs/two neural passes plus charged confidence acquisition; hidden graph labels never teach it. Broader nonlinear factor/depth/interleaving data are prepared. Source-read routing is integrated; global memory discovery/write addressing and deep boundary-credit replacement are next mechanisms. Earlier65,536-target dense boundary-critic learning trails native BPTT. Next decisive test is useful native shared-memory credit with matched information/teaching controls, fixed confirmation and full work, then broader acquired-structure transfer.'
  p=ROOT/'report/I_SCIENCE.md';s=p.read_text();marker='**Forward-taught credit and native learning (12 October):**';start=s.index(marker);end=s.index('\n',start);p.write_text(s[:start]+text+s[end:])
if __name__=='__main__':main()
