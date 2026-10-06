"""Guarded isolated gather backend; original engine and historical kernels intact."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import horizon_token_language_engine as engine
import horizon_token_language as selection
from sleeping_machines.chunk_gather_counterfactual_episodes import token_features

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    receipt=ROOT/'experiments/results/diagnostics/curie_chunk_gather_probe_20261006_v1.json'
    proof=json.loads(receipt.read_text());assert proof['status']=='completed'
    for name,digest in proof['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert proof['integrated']['forward_state_continuation_max_error']==0
    assert proof['integrated']['all_gradient_max_error']<1e-6
    assert proof['integrated']['adamw_update_max_error']==0
    tag=sys.argv[sys.argv.index('--tag')+1]
    output=ROOT/f'experiments/results/token_language/{tag}.json'
    if output.exists():raise FileExistsError(output)
    original=engine.token_features;engine.token_features=token_features
    try:selection.main()
    finally:engine.token_features=original
    result=json.loads(output.read_text())
    result['identity']['source_sha256'].update({name:sha(ROOT/name) for name in ['experiments/gather_token_language.py','sleeping_machines/chunk_gather_counterfactual_episodes.py']})
    result['execution_override']=dict(kind='Credit-window lexical gather only',contract=str(receipt.relative_to(ROOT)),contract_sha256=sha(receipt),scope='Original temporal/sparse engine with state-independent lookup hoisted; no rotation override, altered credit horizon or selection protocol.')
    output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
