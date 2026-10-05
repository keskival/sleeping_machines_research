"""Source-bound selected private-bank utility; guarded development scoring only."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--tag',required=True)
    a=p.parse_args();packet=json.loads(Path(a.manifest).read_text());pins=packet['utility_source_sha256']
    def check():
        for name,digest in pins.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source mismatch: '+name)
    check()
    folder=ROOT/'experiments/results/token_language';tags=packet['fit_tags']
    for tag in tags:
        raw=json.loads((folder/(tag+'.json')).read_text())
        if raw.get('status')!='completed':raise ValueError('Bank fit incomplete: '+tag)
        if raw.get('source_sha256')!=packet['fit_source_sha256']:raise ValueError('Bank fit source mismatch: '+tag)
        if raw.get('bank_recipe',{}).get('bank_policy',{}).get('target_pool')!=raw['args']['pool']:
            raise ValueError('Bank recipe missing/mismatched: '+tag)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if out.exists():raise FileExistsError(out)
    import token_stage_utility as utility
    from aws_private_bank_model import PrivateBankModel
    parent=utility.lab.Model;old_argv=sys.argv
    utility.lab.Model=PrivateBankModel
    sys.argv=['token_stage_utility.py','--tags',*tags,'--output',str(out)]
    try:utility.main()
    finally:utility.lab.Model=parent;sys.argv=old_argv
    check();record=json.loads(out.read_text())
    record['underlying_utility_source_sha256']=record['source_sha256']
    record['source_sha256']=pins
    for row,tag in zip(record['rows'],tags):
        raw=json.loads((folder/(tag+'.json')).read_text())
        row['bank_recipe']=raw['bank_recipe']
        row['available_memory_scalars']=raw['args']['depth']*raw['args']['heads']*raw['args']['pool']*raw['args']['payload']
        row['selected_writes_per_token']=raw['args']['depth']*raw['args']['heads']
        row['scored_keys_per_token']=raw['args']['depth']*raw['args']['heads']*raw['args']['pool']
    record['max_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
