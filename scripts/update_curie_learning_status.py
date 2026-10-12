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
   r=json.loads(native.read_text());assert r['status']=='completed';v={x['arm']:x for x in r['rows']};exact=v['exact'];learned=v['credit']
   text='**Forward-taught credit and native learning (12 October):** The critic uses our native temporal/race family: relationship/query/outcome tokens, independent hard races, persistent addressed state and a linear utility readout. '
   text+=f"Raw native credit improves selection DEV {exact['mean_dev_nll']:.6f}→{learned['mean_dev_nll']:.6f}, fitting {exact['whole_fit_wall_s']:.3f}→{learned['whole_fit_wall_s']:.3f}s. "
   confirmation=ROOT/'experiments/results/causal_induction/curie_causal_race_credit_confirm_20261012_v1_decision.json'
   if confirmation.exists():
    c=json.loads(confirmation.read_text());assert c['status']=='completed';g=c['mean_quality_gains']
    text+=f"Unchanged three-seed/new-graph cohort beats zero-credit and reset-memory in3/3, mean gains {g['zero25']:.6f}/{g['reset_memory']:.6f}; exact teaching in2/3, mean gain {g['exact']:.6f}. Strict broader-transfer gate fails. "
   candidates=[('curie_outcome_bank_20261012_v1_fit','outcome-preserving addressed sharing'),('curie_addressed_bank_20261012_v1_fit','public relative-address sharing'),('curie_task_taught_bank_20261012_v1_fit','task-taught sharing'),('curie_shared_bank_20261012_v2_fit','independent shared-bank prototype')]
   for tag,label in candidates:
    p=ROOT/f'experiments/results/causal_induction/{tag}.json'
    if not p.exists():continue
    c=json.loads(p.read_text());assert c['status']=='completed';rows={x['arm']:x for x in c['rows']};t=rows['reciprocal'];u=rows['untaught']
    text+=f"Current shared-memory development ({label}): reciprocal {t['mean_dev_nll']:.6f}/{t['whole_fit_wall_s']:.3f}s versus matched untaught {u['mean_dev_nll']:.6f}/{u['whole_fit_wall_s']:.3f}s. "
    if t['mean_dev_nll']<learned['mean_dev_nll']:text+='This improves the saved private selection quality; fixed confirmation is next. '
    else:text+='Raw native credit retains the leading selection quality. '
    break
   text+='Shared memory permits independently chosen native reads and receiver writes. Later forward task loss teaches critic content/read operations; auxiliary revealed-loss regression trains its readout. Active relabeling and completion-visibility contracts pass for public relative-address keys. Private four-root affine development: eight TRAIN graphs/two passes plus charged confidence acquisition, fixed DEV weights and causal numerical memory. Teaching gradients cover eight-feedback windows. No efficiency or neural-depth scaling win. Next: useful shared-memory learning versus matched controls with complete work, fixed attribution confirmation, then deeper/larger acquired-structure tests; global destination learning and deep boundary-credit replacement are integration gaps.'
  world=ROOT/'experiments/results/causal_induction/curie_world_credit_20261012_v2_decision.json'
  if world.exists():
   w=json.loads(world.read_text());assert w['status']=='completed';v={r['arm']:r for r in w['rows']}
   text='**Forward-taught credit and native learning (12 October):** Native outcome-law credit uses one connected temporal world query and revealed-outcome update, independently selected shared-memory access, and coherent loss projections across alternative routes. '
   text+=f"Common-source causal reuse: learned credit NLL {v['learned']['pooled_nll']:.6f}, exact observed-label {v['exact']['pooled_nll']:.6f}, fixed-prior blend {v['prior']['pooled_nll']:.6f}. "
   text+='The development gate passes; unchanged fresh-actor/fresh-system confirmation is next. ' if w['development_gate'] else 'The development gate fails; diagnose conditional outcome information and credit sensitivity before scaling. '
   text+=f"World NLL {v['learned']['world_nll']:.6f}; gain versus causal global-frequency diagnostic {w['learned_world_gain_vs_frequency']:+.6f}. Complete source-plus-reuse stage wall: "+', '.join(f"{k} {v[k]['complete_acquisition_and_reuse_stage_wall_s']:.3f}s" for k in ['learned','exact','prior'])+'. '
   text+='Scope: one actor,16nonlinear TRAIN systems/2048unique source outcomes and4096charged presentations,8fresh8/16-factor4/8-generating-depth systems/1024prequential targets per arm. Common actor/world/Adam fingerprints; all arms train the world and expose identical shared-memory capability. Learned/prior half blends intentionally change conditional credit means; source-choice credit only, eight-feedback gradients, no neural-depth or efficiency win. Previous raw-native affine evidence and shared-bank development remain in the completed appendix. The next architectural target is calibrated causal-response transfer, with compact task projections from rich native persistent state.'
  p=ROOT/'report/I_SCIENCE.md';s=p.read_text();marker='**Forward-taught credit and native learning (12 October):**';start=s.index(marker);end=s.index('\n',start);p.write_text(s[:start]+text+s[end:])
if __name__=='__main__':main()
