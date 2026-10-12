"""Refresh only Curie's current science/status paragraph from completed stages."""
import fcntl,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);path=ROOT/'experiments/results/credit/curie_native_operator_event_20261012_v1_fit.json';label='operator-conditioned recursive repair'
  if not path.exists():path=ROOT/'experiments/results/credit/curie_native_cohort_event_20261012_v1_s99913_fit.json';label='complete cohort/depth stage'
  r=json.loads(path.read_text());assert r['status']=='completed';comparisons=[]
  for row in r['rows']:
   arms=row['arms'];comparisons.append(f"depth{row['depth']}: online DEV{arms['online']['final_dev_nll']:.6f}, heads{arms['heads']['final_dev_nll']:.6f}, nativeAdamW{arms['bp_adam']['final_dev_nll']:.6f}; wholeCPU{arms['online']['whole_learning_wall_s']:.3f}/{arms['bp_adam']['whole_learning_wall_s']:.3f}s")
  loss_depths=[row['depth'] for row in r['rows'] if row['online_gain_vs_adam']<=0];win_depths=[row['depth'] for row in r['rows'] if row['online_gain_vs_adam']>0]
  verdict=(' Online trails nativeAdamW at depths'+str(loss_depths)+'.' if loss_depths else '')+(' Single-seed development quality wins versus nativeAdamW at depths'+str(win_depths)+'.' if win_depths else '')
  text='The completed '+label+' records '+ '; '.join(comparisons)+'.'+verdict+' Exact acquired-core/head and frozen-credit controls are retained; full per-arm acquisition/teacher/optimizer/scoring work is charged. The appendix states each depth result. Reused TaxiDEV is development evidence; no TEST or hardware claim.'
  p=ROOT/'report/I_SCIENCE.md';s=p.read_text();start=s.index('**Forward-taught credit and native learning (12 October):**');end=s.index('\n',start);para=s[start:end];begin=para.find('Local variants stop;')
  if begin>=0:
   finish=para.find(' These native comparisons',begin);finish=len(para) if finish<0 else finish;para=para[:begin]+text+para[finish:]
  else:para+=' '+text
  para=para.replace('These native comparisons reuse32TaxiDEV sequences/512targets','The local stages reuse32TaxiDEV sequences/512targets; the larger stages use128sequences/2048targets')
  p.write_text(s[:start]+para+s[end:])
  p=ROOT/'experiments/HANDOFF.md';s=p.read_text();end=s.index('\n\n');p.write_text('**Curie current native learning stage complete — 12 October.** '+text+' Source result: '+str(path.relative_to(ROOT))+'. Prior local event optimizer loses7/8 exact matchedheads; previouscrossconfigurationdeepbenefitcorrected. All evidence/sourcehashes preserved; immutable completeddrivers. See conditional continuation decisions and currentpublication receipts under.git. Operator-conditionedmessage rationale and primitive/futuregradient contracts are B1development; retain actor temporal/addressed/silence mechanisms, densewrites/independentepisoderesets/nooptionalforwardroutecredit gaps. NoTEST/PAM/externaldense/varianceproxy jobs. Preserve main/otherhost/unrelatedchanges.\n'+s[end:])
if __name__=='__main__':main()
