"""Current Curie market findings and canonical report publication."""
import argparse,fcntl,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 path=ROOT/'experiments/results/market/curie_b10_exact_future_20261012_v1_fit.json';r=json.loads(path.read_text());assert r['status']=='completed'
 for name,h in r['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 v={x['arm']:x for x in r['rows']}
 paragraph=f"**Real-market online credit learning (12 October):** Native prospective guidance with full exact actor credit wins the five-arm development quality comparison on4096BTCUSDT Aug25forecasts. Online NLL{v['online']['nll']:.6f}, ordinary online BPTT{v['bptt']['nll']:.6f}, frozen guidance{v['frozen']['nll']:.6f}, local-only{v['zero']['nll']:.6f}, fully frozen{v['fully_frozen']['nll']:.6f}. Online gains BPTT:+{r['online_gain']['bptt']:.6f}, frozen:+{r['online_gain']['frozen']:.6f}nats/event. Common acquired state/optimizers; Aug24TRAIN continuation2048targets. Reuse online{v['online']['wall_s']:.3f}s/BPTT{v['bptt']['wall_s']:.3f}s, shared acquisition{r['acquisition']['wall_s']:.3f}s and original predictor development5750.687s. Full18164parameter saved actor,33232parameter native hard-race critic; AWS causal history, forward-response inputs and actual next-window clipped-Adam teaching integrated. One initialization/reused DEV, no TEST/profit/hardware claim. Small quality win with higher learning cost; BPTT and frozen guidance lead in the second half. Earlier sampled-anchor native credit improves frozen but trails BPTT; its zero-predictor residual-audit arm was not local-only. Longer chronological development follows before confirmation."
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  p=ROOT/'report/I_SCIENCE.md';s=p.read_text();marker='**Real-market online credit learning (12 October):**'
  if marker in s:
   start=s.index(marker);end=s.find('\n\n',start);s=s[:start]+paragraph+s[end:]
  else:
   anchor='**Forward-taught credit and native learning (12 October):**';start=s.index(anchor);end=s.find('\n\n',start);s=s[:end]+'\n\n'+paragraph+s[end:]
  p.write_text(s)
  p=ROOT/'experiments/HANDOFF.md';s=p.read_text();marker='**Curie B10 native-credit test-time learning — 12 October.**';start=s.index(marker);end=s.find('\n\n',start);s=s[:start]+paragraph+' Protocol market/NATIVE_EXACT_FUTURE_PROTOCOL.md. OneCPU/RSS2.2GB/9GiBfloor/2GiBcgroupreserve.'+s[end:];p.write_text(s)
 subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/build_credit_report_parts.py','--tag',a.tag],cwd=ROOT,check=True)
if __name__=='__main__':main()
