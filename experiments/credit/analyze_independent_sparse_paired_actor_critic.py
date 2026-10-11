"""Completed repaired-stage analysis, preserving the original analysis driver."""
import hashlib
import json
from pathlib import Path
import sys
import analyze_sparse_paired_actor_critic as analysis
from independent_sparse_paired_actor_critic import TAG

if __name__=='__main__':
    analysis.PREFIX=TAG;analysis.main()
    tag=sys.argv[sys.argv.index('--tag')+1]
    out=analysis.ROOT/f'experiments/results/credit/{tag}.json'
    d=json.loads(out.read_text())
    d['source_sha256'][str(Path(__file__).relative_to(analysis.ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    d['protocol_repair']='Independent cloned histories/buffers/optimizer states, with validated exact cached continuation replay; original128 acquisition cost retained.'
    out.write_text(json.dumps(d,indent=2)+'\n')
