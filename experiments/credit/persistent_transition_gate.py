"""Run persistent tape-transition contracts inside the admitted gym slot."""
import argparse,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pytest

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();start=time.monotonic()
 assert pytest.main(['-q','tests/test_persistent_tape_transition.py'])==0
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
 out.write_text(json.dumps(dict(status='completed',battle='R1',stage='persistent native tape transition gate',gate='PASS',elapsed_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
if __name__=='__main__':main()
