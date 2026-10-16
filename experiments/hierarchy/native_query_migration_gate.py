"""R1 checkpoint migration contract; no data acquisition or DEV/TEST."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import torch,pytest

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);began=time.monotonic()
    assert pytest.main(['-q','tests/test_native_query_migration.py'])==0
    names=['experiments/native_query_migration.py','tests/test_native_query_migration.py','sleeping_machines/packed_queryable_local_deep_race.py']
    result=dict(status='completed',battle='R1',stage='native_query_migration_gate',gate='PASS',
        wall_s=time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        scope='Tiny native full-learner contract:shared parameters/Adam states and persistent history preserved;bitwise serialized continuation including query lineage and both clock generators. No quality result or acquisition.')
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
