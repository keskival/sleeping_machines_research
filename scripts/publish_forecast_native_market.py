"""Publish a completed forecast-response market comparison with full cost."""
import argparse,fcntl,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 result='experiments/results/market/curie_b10_forecast_response_20261012_v1_fit.json';r=json.loads((ROOT/result).read_text());assert r['status']=='completed'
 for p,h in r['source_sha256'].items():assert sha(ROOT/p)==h,p
 rows={x['arm']:x for x in r['rows']};online=rows['online'];rate=online['learning_rate_multipliers'];verdict='passes' if r['quality_gate'] else 'fails'
 paragraph=f"**Predictive-law-conditioned native credit (12 October, completed):** Added factual/counterfactual clock-law inputs to the same native event critic through a zero-initialized162-to16 adapter. On4096Aug25development forecasts the seven-arm quality gate {verdict}: online NLL{online['nll']:.6f}, ordinary BPTT{rows['bptt']['nll']:.6f}, frozen guidance{rows['frozen']['nll']:.6f}, online field-only{rows['field_only']['nll']:.6f}, response-masked online{rows['response_masked']['nll']:.6f}. Online gains BPTT{r['online_gain']['bptt']:+.6f}/frozen{r['online_gain']['frozen']:+.6f}/masked{r['online_gain']['response_masked']:+.6f}nats/event. Mean Adam multiplier{sum(rate)/len(rate):.4f}; adapter parameter change{online['forecast_adapter_max_change']:.6g}. Online reuse{online['wall_s']:.3f}s versus BPTT{rows['bptt']['wall_s']:.3f}s; shared acquisition{r['acquisition']['wall_s']:.3f}s plus original5750.687s fitting charged to every arm. Full exact32-event actor credit retained: information/quality experiment, not backprop replacement or efficiency win. Common acquired state/capacity; masked arm tests deployment-input usefulness, not independent acquisition. One initialization/reused DEV; sealedTEST untouched."
 with open('/tmp/aws-language-publication.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True).strip()
  for name in ['experiments/HANDOFF.md','report/I_SCIENCE.md']:
   p=ROOT/name;s=p.read_text();marker='**Predictive-law-conditioned native credit (12 October, completed):**'
   if marker in s:
    start=s.index(marker);end=s.find('\n\n',start);s=s[:start]+paragraph+s[end:]
   else:s=paragraph+'\n\n'+s if name.endswith('HANDOFF.md') else s+'\n\n'+paragraph+'\n'
   p.write_text(s)
  p=ROOT/'experiments/market/B10_MARKET.md';s=p.read_text();s+='\n### Predictive-law input comparison (12 October)\n\n'+paragraph+'\n\n| Rule | NLL | Reuse seconds |\n|---|---:|---:|\n'
  for x in r['rows']:s+=f"| {x['arm']} | {x['nll']:.6f} | {x['wall_s']:.3f} |\n"
  p.write_text(s)
  p=ROOT/'investment/brief/evidence_brief.tex';s=p.read_text();old='A subsequent optimizer-control experiment corrected\nunstable update sizes, but its learned guidance still trails ordinary online\nlearning by 0.000408 nats per transaction.'
  direction='improves on' if r['online_gain']['bptt']>0 else 'trails'
  new=f"The latest forecast-response-conditioned guidance {direction}\nordinary online learning by {abs(r['online_gain']['bptt']):.6f} nats per transaction\non 4,096 development forecasts. It still runs exact backprop and adds learning\nwork; this measures prediction quality, not a learning-efficiency gain."
  assert old in s;s=s.replace(old,new);p.write_text(s)
 python=str(ROOT/'.venv-docker/bin/python')
 subprocess.run([python,'scripts/build_credit_report_parts.py','--tag',a.tag],cwd=ROOT,check=True)
 subprocess.run([python,'scripts/build_forward_credit_evidence_brief.py','--tag',a.tag],cwd=ROOT,check=True)
 report=json.loads((ROOT/f'.git/credit-report-preview/{a.tag}/publication.json').read_text());brief=json.loads((ROOT/f'.git/evidence-brief-preview/{a.tag}/publication.json').read_text())
 with open('/tmp/aws-language-publication.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  paths=[result,'experiments/HANDOFF.md','experiments/market/B10_MARKET.md','investment/brief/evidence_brief.tex','investment/forward_credit_evidence_20261011.json','investment/sleeping_machines_evidence_brief.pdf','report/sleeping_machines_status.pdf','report/sleeping_machines_model_family.pdf']
  for p,h in report['artifact_sha256'].items():assert sha(ROOT/p)==h,p;paths.append(p)
  assert sha(ROOT/brief['artifact'])==brief['sha256']
  subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish native forecast-response market comparison and complete cost','--',*paths],cwd=ROOT,check=True)
 print(paragraph,flush=True)
if __name__=='__main__':main()
