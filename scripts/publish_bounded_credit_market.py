"""Publish completed bounded credit, exact-execution repair and primitive support."""
import argparse,fcntl,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 result='experiments/results/market/curie_b10_bounded_feedback_20261012_v1_fit.json';r=json.loads((ROOT/result).read_text());assert r['status']=='completed'
 for p,h in r['source_sha256'].items():assert sha(ROOT/p)==h,p
 rows={x['arm']:x for x in r['rows']};online=rows['online'];d=json.loads((ROOT/'experiments/results/market/curie_b10_joint_bounded_20261012_v1_decision.json').read_text());assert d['block_score_max_error']==0
 paragraph=f"**Bounded native market credit (12 October, completed):** One aggregate timed receipt per32-target update plus future teaching every16updates reduces deployment overhead. On4096Aug25DEV forecasts online NLL{online['nll']:.6f}, ordinary BPTT{rows['bptt']['nll']:.6f}, frozen guidance{rows['frozen']['nll']:.6f}, local-only{rows['zero']['nll']:.6f}, input-masked online{rows['response_masked']['nll']:.6f}. Online gains BPTT{r['online_gain']['bptt']:+.6f}/frozen{r['online_gain']['frozen']:+.6f}/masked{r['online_gain']['response_masked']:+.6f}nats/event; seven-arm quality gate fails. Reuse online{online['wall_s']:.3f}s/BPTT{rows['bptt']['wall_s']:.3f}s, common acquisition{r['acquisition']['wall_s']:.3f}s plus original5750.687s predictor fitting. Exact actor credit retained; one reused DEV stream/initialization, TEST untouched. Original512-target optimistic cost gate passed0.373<0.807s. Exact joint parameter/cotangent extraction then reproduces all source/512block scores with0error, reduces BP/learned reuse wall27.1%/22.5%, and tightens the remaining transport/proposal budget to{d['remaining_transport_proposal_s']:.3f}s against added learned work{d['incremental_loop_s']:.3f}s: replacement budget fails. This is an exact implementation improvement, not a learned-credit efficiency win. Next integration targets inside-history graph work using exact local-clock eligibility plus learned cross-leg credit."
 with open('/tmp/aws-language-publication.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True).strip()
  for name in ['experiments/HANDOFF.md','report/I_SCIENCE.md']:
   p=ROOT/name;s=p.read_text();marker='**Bounded native market credit (12 October, completed):**'
   if marker in s:
    start=s.index(marker);end=s.find('\n\n',start);s=s[:start]+paragraph+s[end:]
   elif name.endswith('HANDOFF.md'):
    # Keep the AWS leading paragraph stable; update our front below it.
    end=s.find('\n\n');s=s[:end]+'\n\n'+paragraph+s[end:]
   else:s+='\n\n'+paragraph+'\n'
   s=s.replace('Cost512DEV running tmuxcurie_market_bounded_20261012','Cost512DEV and4096DEV comparison completed; controller tmuxcurie_market_bounded_20261012 finished')
   p.write_text(s)
  p=ROOT/'experiments/market/B10_MARKET.md';s=p.read_text();s+='\n### Bounded credit and exact execution repair (12 October)\n\n'+paragraph+'\n\n| Rule | NLL | Reuse seconds |\n|---|---:|---:|\n'
  for x in r['rows']:s+=f"| {x['arm']} | {x['nll']:.6f} | {x['wall_s']:.3f} |\n"
  p.write_text(s)
  p=ROOT/'investment/brief/evidence_brief.tex';s=p.read_text();old='The latest forecast-response-conditioned guidance trails\nordinary online learning by 0.000388 nats per transaction\non 4,096 development forecasts. It still runs exact backprop and adds learning\nwork; this measures prediction quality, not a learning-efficiency gain.'
  new='The latest bounded learning controller trails ordinary online learning by\n0.000047 nats per transaction on 4,096 development forecasts. It improves on\nfrozen guidance by 0.000026, with learning time 26.5 versus 23.2 seconds.\nIt still runs exact backprop; this measures prediction quality, not a\nlearning-efficiency gain.'
  assert old in s;s=s.replace(old,new);p.write_text(s)
 python=str(ROOT/'.venv-docker/bin/python')
 subprocess.run([python,'scripts/build_credit_report_parts.py','--tag',a.tag],cwd=ROOT,check=True)
 subprocess.run([python,'scripts/build_forward_credit_evidence_brief.py','--tag',a.tag],cwd=ROOT,check=True)
 print(paragraph,flush=True)
if __name__=='__main__':main()
