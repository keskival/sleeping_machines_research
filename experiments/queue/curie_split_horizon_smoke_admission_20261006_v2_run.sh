#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 200); do
 if ! tmux has-session -t curie_split_horizon_driver_contracts_20261006_v2 2>/dev/null; then break; fi
 sleep 30
done
if tmux has-session -t curie_split_horizon_driver_contracts_20261006_v2 2>/dev/null; then exit 75; fi
python3 - <<'CHECK'
import hashlib,json
from pathlib import Path
pins={'experiments/split_horizon_token_language_engine.py': '6085552561e26c09ef0be68791d016db40cde2b8ef6f6ec1693ac91b4e1e20f7', 'experiments/split_horizon_token_language.py': 'f2ba7a73f97fb3e523e0a0d6e2887c142b1688f2ea170a89d8243b76f8376678', 'experiments/split_horizon_state_contracts.py': 'cf23d809da8a1fef25fedcc03166a81e223713c6ad389bf9b8787dda33468048', 'experiments/split_horizon_driver_contracts_v2.py': 'ae4b79c8948171a7639a46197ee94ea50055ccf2c08f4a7d0d4dd004117a99f4'}
for path,digest in pins.items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
folder=Path('experiments/results/diagnostics')
r=json.loads((folder/'curie_split_horizon_state_contracts_20261006_v1.json').read_text())
assert r['status']=='completed'
assert all(r[k] for k in ('numeric_state_parity','feature_parity','route_rng_parity','factual_gradient_contract'))
r=json.loads((folder/'curie_split_horizon_driver_contracts_20261006_v2.json').read_text())
assert r['status']=='completed' and r['producer_sha256']==pins['experiments/split_horizon_driver_contracts_v2.py']
assert all(r[k] for k in ('diagonal_parity','off_diagonal_resume','causal_endpoints','importance_weight_contract'))
CHECK
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=180 bash experiments/queue/run_safe.sh experiments/queue/curie_split_horizon_2k_f64_a16_s6_20261006_v1.txt
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=180 bash experiments/queue/run_safe.sh experiments/queue/curie_split_horizon_2k_f16_a64_s6_20261006_v1.txt
