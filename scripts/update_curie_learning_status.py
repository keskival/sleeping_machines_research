"""Refresh Curie's current science paragraph from completed causal/native stages."""
import fcntl,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  text='**Forward-taught credit and native learning (12 October):** The generated-credit full-cohort depth8/16/32 stage trails native AdamW. Operator-aware updates and bounded actual-step corrections retain exact current-event adjoints; bounded online credit beats frozen credit at depths8/16 but trails reset credit at both and uses more CPU. These are development results on reused Taxi DEV, with complete acquisition/teacher/optimizer/scoring CPU charged. '
  p=ROOT/'experiments/results/credit/curie_market_credit_20261012_v1_fit.json'
  if p.exists():
   r=json.loads(p.read_text());assert r['status']=='completed';v={x['arm']:x for x in r['rows']}
   text+=f"Chronological market TRAIN continuation over {v['online']['targets']} targets records online-credit NLL {v['online']['nll']:.6f}, frozen-credit {v['frozen']['nll']:.6f}, zero-audit {v['zero']['nll']:.6f}, native BPTT {v['bptt']['nll']:.6f}. Online credit loses to all three; reuse CPU is {v['online']['wall_s']:.3f}s versus BPTT {v['bptt']['wall_s']:.3f}s, with shared acquisition {r['acquisition']['wall_s']:.3f}s. "
  p=ROOT/'experiments/results/credit/curie_market_hidden_shift_20261012_v1_fit.json'
  if p.exists():
   r=json.loads(p.read_text());assert r['status']=='completed';text+='Hidden dependency reversal with identical population marginals: '+', '.join(f"{x['arm']} NLL {x['nll']:.6f}" for x in r['rows'])+'. Synthetic marks on TRAIN timestamps make this a mechanism diagnostic. '
  for tag,label in [('curie_market_future_credit_20261012_v1_fit','Future-loss interventions on real TRAIN'),('curie_market_future_shift_20261012_v1_fit','Future-loss interventions under hidden conditional reversal')]:
   p=ROOT/f'experiments/results/credit/{tag}.json'
   if p.exists():
    r=json.loads(p.read_text());assert r['status']=='completed';text+=label+': '+', '.join(f"{x['arm']} NLL {x['nll']:.6f} / {x['wall_s']:.3f}s" for x in r['rows'])+'. Shared acquisition is charged separately; Gaussian innovations are not paired, one-seed development. '
  text+='Next decisive test trains interventions for future predictive utility rather than cotangent imitation, retaining causal state and matched frozen-credit/ordinary-online controls. The current native local/audit work is dense; these results make no sparse hardware or sealed TEST claim.'
  p=ROOT/'report/I_SCIENCE.md';s=p.read_text();marker='**Forward-taught credit and native learning (12 October):**';start=s.index(marker);end=s.index('\n',start);p.write_text(s[:start]+text+s[end:])
if __name__=='__main__':main()
