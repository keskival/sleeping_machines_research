"""Current public-win headline;means define wins,seed/split counts are separate."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 inputs={}
 def load(name):
  p=ROOT/name;inputs[name]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
 p=load('report/public_wins_headline_evidence_20261010_v4.json');assert p['status']=='completed'
 inputs.update(p['input_sha256']);rows=[]
 def add(key,label,values,reference,higher,metric,protocol,scope):
  mean=statistics.mean(values);assert (mean>reference if higher else mean<reference),key
  rows.append(dict(key=key,label=label,values=values,mean=mean,reference=reference,higher_is_better=higher,metric=metric,protocol=protocol,ahead=sum(x>reference if higher else x<reference for x in values),runs=len(values),scope=scope))
 for k,d in p['easytpp'].items():add('easytpp_'+k,'EasyTPP '+k,d['ours']['values'],d['published']['ll'],True,'nats/event','5 seeds','Dataset-specific completed native member;best published single-model reference.')
 clinical=load('experiments/results/irts/b2_final_p19_v3_summary.json')
 add('p19','P19 sepsis',[s['auprc'] for s in clinical['per_split']],p['p19']['published']['auprc'],True,'AUPRC','5 official splits','Completed clinical protocol.')
 add('pam','PAM wearables',p['pam']['acc']['values'],p['pam']['published']['acc'],True,'accuracy','5 official splits','Completed clinical protocol.')
 add('tgb_trade','TGB trade',p['tgbn_trade']['ndcg']['values'],p['tgbn_trade']['published']['ndcg'],True,'NDCG@10','3 seeds','Preregistered official TGB evaluation.')
 add('tgb_wiki','TGB wiki',p['tgbl_wiki']['mrr']['values'],p['tgbl_wiki']['published']['test'],True,'MRR','3 seeds','Preregistered official TGB evaluation.')
 established=dict(wins=len(rows),ahead=sum(r['ahead'] for r in rows),runs=sum(r['runs'] for r in rows));assert established==dict(wins=9,ahead=40,runs=41)
 for k,label in [('wikipedia','Wikipedia'),('github','Github'),('stackoverflow','Stack Overflow')]:
  d=p['b4_'+k+'_developed']
  add('b4_'+k,'B4 '+label,[r['total'] for r in d['test_splits']],d['best_single_published'],False,'NLL/sequence','5 official splits','Developed native variant selected by aggregate validation;TEST once,recorded-density scoring. Full work not audited.')
  rows[-1].update(composite_target=d['composite_target'],beats_composite=d['beats_composite_target'],test_se=d['test_total_se'])
 d=p['b4_lastfm'];add('b4_lastfm','B4 LastFM',d['total']['values'],d['best_single'],False,'NLL/sequence','5 official splits','Frozen one-configuration transfer,recorded-density scoring. Full work not audited.')
 rows[-1].update(composite_target=d['bar'],beats_composite=d['total']['mean']<d['bar'],test_se=d['total']['se'])
 result=dict(status='completed',date='2026-10-16',wins=len(rows),ahead=sum(r['ahead'] for r in rows),runs=sum(r['runs'] for r in rows),established=established,rows=rows,
 scope='13 winning dataset/protocol entries against best published single-model mean. Across those entries53/61 individual seed/split scores are ahead;official splits are not independent trials and this count is not pooled statistical confidence. B4 Wikipedia/Github also beat stronger composite targets;Stack Overflow/LastFM miss those composites. FAS own-benchmark confirmation and B10 market confirmation are additional evidence,excluded from this public-reference count. Other developing benchmark gaps remain in the complete record.',input_sha256=inputs)
 assert (result['wins'],result['ahead'],result['runs'])==(13,53,61)
 (ROOT/'investment/public_benchmark_headlines.json').write_text(json.dumps(result,indent=2)+'\n')
 print('13 public benchmark wins;53/61 seed/split scores ahead;B4 composite wins2/4.')
if __name__=='__main__':main()
