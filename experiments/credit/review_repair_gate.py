"""Bounded R1 integrated repair contracts, executed only through the gym."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,pytest

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--tests',nargs='+',default=['tests/test_review_repairs.py']);a=p.parse_args();torch.set_num_threads(1);start=time.monotonic()
    names=[str(p.relative_to(ROOT)) for p in (ROOT/'sleeping_machines').glob('*.py')]+a.tests+['experiments/credit/review_repair_gate.py','experiments/r1_rmr_writer_input_credit.py','experiments/r1_rmr.py','experiments/r1_cerl.py','experiments/native_slot_memory.py','experiments/r1_deep_keyed_lm.py','experiments/r1_local_native_deep.py']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    assert pytest.main(['-q',*a.tests])==0
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    out.write_text(json.dumps(dict(status='completed',battle='R1',stage='review repair integrated numerical gate',gate='PASS',source_sha256=hashes,elapsed_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Numerical repair contracts only, no quality or benchmark win.'),indent=2)+'\n')
if __name__=='__main__':main()
