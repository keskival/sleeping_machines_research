"""Publish and commit a completed longer chronological market stage."""
import argparse,fcntl,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 p=ROOT/'experiments/results/market/curie_b10_longer_native_20261012_v1_fit.json';r=json.loads(p.read_text());assert r['status']=='completed' and r['prefix_replay_max_error']<1e-10
 for path,h in r['source_sha256'].items():assert sha(ROOT/path)==h,path
 rows={v['arm']:v for v in r['rows']};verdict='passes' if r['quality_gate'] else 'fails'
 paragraph=f"**Longer real-market online credit (12 October):** Unchanged native prospective guidance on16384Aug25DEV forecasts {verdict} the five-control quality gate. Online NLL{rows['online']['nll']:.6f}, BPTT{rows['bptt']['nll']:.6f}, frozen guidance{rows['frozen']['nll']:.6f}, local-only{rows['zero']['nll']:.6f}, fully-frozen{rows['fully_frozen']['nll']:.6f}; learned gains BPTT{r['online_gain']['bptt']:+.6f}/frozen{r['online_gain']['frozen']:+.6f}nats/event. All first4096forecast losses replay within{r['prefix_replay_max_error']:.2e}. Complete online continuation{r['acquisition']['wall_s']+rows['online']['wall_s']:.3f}s, BPTT{r['acquisition']['wall_s']+rows['bptt']['wall_s']:.3f}s, plus shared5750.687s original predictor development. One initialization/reused development stream, no sealedTEST, profitability, measured regime shift or hardware claim. The completed4096target small quality win remains scoped to that prefix; the longer result measures its persistence."
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for path in ['experiments/HANDOFF.md','report/I_SCIENCE.md']:
   dest=ROOT/path;s=dest.read_text();marker='**Longer real-market online credit (12 October):**'
   if marker in s:
    start=s.index(marker);end=s.find('\n\n',start);s=s[:start]+paragraph+s[end:]
   elif path.endswith('HANDOFF.md'):s=paragraph+'\n\n'+s
   else:
    marker='**Real-market online credit learning (12 October):**';start=s.index(marker);end=s.find('\n\n',start);s=s[:end]+'\n\n'+paragraph+s[end:]
   dest.write_text(s)
  dest=ROOT/'experiments/market/B10_MARKET.md';s=dest.read_text();s+='\n### Completed longer chronological stage (12 October)\n\n'+paragraph+'\n\n| Rule | First4096 | Second4096 | Third4096 | Last4096 |\n|---|---:|---:|---:|---:|\n'
  for row in r['rows']:s+='| '+row['arm']+' | '+' | '.join(f'{v:.6f}' for v in row['quarter_nll'])+' |\n'
  dest.write_text(s)
  dest=ROOT/'investment/brief/evidence_brief.tex';s=dest.read_text();anchor='This is a small prediction-quality advantage, with a higher\nlearning cost. It measures transaction forecasting, not trading profits.'
  text=f"This is a small prediction-quality advantage, with a higher\nlearning cost. A completed extension to 16,384 forecasts gives learned guidance\n{r['online_gain']['bptt']:+.6f} nats per transaction versus ordinary online learning\n(positive means better). It measures transaction forecasting, not trading profits."
  assert anchor in s;s=s.replace(anchor,text);dest.write_text(s)
 python=str(ROOT/'.venv-docker/bin/python')
 subprocess.run([python,'scripts/build_credit_report_parts.py','--tag',a.tag],cwd=ROOT,check=True)
 subprocess.run([python,'scripts/build_forward_credit_evidence_brief.py','--tag',a.tag],cwd=ROOT,check=True)
 report=json.loads((ROOT/f'.git/credit-report-preview/{a.tag}/publication.json').read_text());brief=json.loads((ROOT/f'.git/evidence-brief-preview/{a.tag}/publication.json').read_text())
 paths=['experiments/results/market/curie_b10_longer_native_20261012_v1_fit.json','experiments/HANDOFF.md','experiments/market/B10_MARKET.md','investment/brief/evidence_brief.tex','investment/forward_credit_evidence_20261011.json','investment/sleeping_machines_evidence_brief.pdf']
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert sha(ROOT/brief['artifact'])==brief['sha256']
  assert sha(ROOT/'investment/brief/evidence_brief.tex')==brief['source_sha256']
  assert sha(ROOT/'investment/forward_credit_evidence_20261011.json')==brief['packet_sha256']
  for path,h in report['artifact_sha256'].items():assert sha(ROOT/path)==h,path;paths.append(path)
  paths.extend(['report/sleeping_machines_status.pdf','report/sleeping_machines_model_family.pdf','report/figures/native_controller_depth_gain.svg','report/figures/future_credit_variance.svg'])
  paths=[p for p in paths if (ROOT/p).exists()]
  assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
  subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish completed longer native market online-credit comparison','--',*paths],cwd=ROOT,check=True)
 print(paragraph,flush=True)
if __name__=='__main__':main()
