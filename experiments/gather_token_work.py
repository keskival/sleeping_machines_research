"""Full executed gather fitting/inference ledger; reuse the existing audit boundary."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import token_stage_work as audit
from sleeping_machines.chunk_gather_counterfactual_episodes import token_features

def main():
    control=Path(sys.argv[sys.argv.index('--control')+1]);saved=json.loads(control.read_text())
    assert saved['status']=='completed' and saved['execution_override']['kind']=='Credit-window lexical gather only'
    original=audit.engine.token_features;audit.engine.token_features=token_features
    try:audit.main()
    finally:audit.engine.token_features=original
    out=Path(sys.argv[sys.argv.index('--output')+1]);record=json.loads(out.read_text())
    record['execution_override']=saved['execution_override']
    record['wrapper_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record['gather_source_sha256']=hashlib.sha256((ROOT/'sleeping_machines/chunk_gather_counterfactual_episodes.py').read_bytes()).hexdigest()
    out.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
