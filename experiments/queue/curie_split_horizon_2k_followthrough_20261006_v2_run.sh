#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 200); do
 if ! tmux has-session -t curie_split_horizon_smoke_admission_20261006_v2 2>/dev/null; then break; fi
 sleep 30
done
if tmux has-session -t curie_split_horizon_smoke_admission_20261006_v2 2>/dev/null; then exit 75; fi
python3 - <<'CHECK'
import hashlib,json
from pathlib import Path
pins={'experiments/split_horizon_token_stage_work.py': '3af49e28a5937d5d7af77f37a7ba54957e6478be21230e4e553f792dd886bfb7', 'experiments/split_horizon_token_stage_utility.py': 'c9b01d4adaf4eff12dfe1a2f2e5231864b985bfd2e320e8ff3266eff38ab79a0'}
tags=['curie_split_horizon_2k_f64_a16_s6_20261006_v1', 'curie_split_horizon_2k_f16_a64_s6_20261006_v1']
for path,digest in pins.items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
for tag in tags:
 path=Path('experiments/results/token_language')/(tag+'.json')
 r=json.loads(path.read_text());assert r['status']=='completed' and r['args']['steps']==8 and r['presentations_total']==4080
 selected=json.loads(path.with_suffix('.selection.json').read_text());assert selected['status']=='completed'
 for source,digest in r['identity']['source_sha256'].items():
  assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest,source
CHECK
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=300 bash experiments/queue/run_safe.sh experiments/queue/curie_split_horizon_2k_utility_20261006_v1.txt
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_split_horizon_2k_f64_a16_work_20261006_v1.txt
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_split_horizon_2k_f16_a64_work_20261006_v1.txt
