"""G-HIER source-bound numerical contracts, no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    start=time.monotonic()
    subprocess.run([sys.executable,'-m','pytest','tests/test_spatial_race_hierarchy.py','-q'],cwd=ROOT,check=True)
    names=['experiments/hierarchy/contracts.py','sleeping_machines/spatial_race_hierarchy.py','tests/test_spatial_race_hierarchy.py']
    d=dict(status='completed',battle='G-HIER',stage='contracts',tests=8,
           scope='32 composition blocks; global feature gradients; all-block local key/value credit; exact local forced replacement; spatial sensitivity/locality; owned RNG. No fit/semantic claim.',
           process_wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
           source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names})
    out=ROOT/f'experiments/results/hierarchy/{a.tag}.json';out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(d,f,indent=2);f.write('\n')

if __name__=='__main__':main()
