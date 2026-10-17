"""Source-bound numerical witnesses for the R1 architecture review."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pytest

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();start=time.monotonic()
    names=['tests/test_architecture_review_witnesses.py','experiments/credit/architecture_review_gate.py']
    names += [str(p.relative_to(ROOT)) for p in (ROOT/'sleeping_machines').glob('*.py')]
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    assert pytest.main(['-q','tests/test_architecture_review_witnesses.py'])==0
    result=dict(status='completed',battle='R1',stage='architecture review mathematical witnesses',gate='PASS',source_sha256=hashes,elapsed_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Finite analytic and native-kernel derivative witnesses, no training or benchmark claim. Missing direct-path credit is distinct from the integrated query-credit paths.')
    (ROOT/f'experiments/results/diagnostics/{a.tag}.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
