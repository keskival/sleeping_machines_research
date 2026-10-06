#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 400); do
 if ! tmux has-session -t curie_split_horizon_64k_followthrough_20261006_v1 2>/dev/null; then break; fi
 sleep 30
done
if tmux has-session -t curie_split_horizon_64k_followthrough_20261006_v1 2>/dev/null; then exit 75; fi
python3 - <<'CHECK'
import hashlib,json
from pathlib import Path
pins={'experiments/horizon_token_language_engine.py': '39b11a75915266f609ca968c14a8953d7aba4403646aa5dcc1812afa9e5cf5f8', 'sleeping_machines/sparse_counterfactual_episodes.py': '7fd2cbd724ed60843a7a94b8d14ec0baf9bd9cf54af2d8b22620f0931aa824d8', 'sleeping_machines/sparse_counterfactual_layer.py': '41a290063110c8b9a5919c4edf0940890097617a431bbcf8f3f0e8b28ecc1f37', 'sleeping_machines/sparse_training.py': 'f259279d2a2af58defe1717a20b7d4fd6639b11095513a323ed8e1b42c3e036f', 'sleeping_machines/packed_token_core.py': '9d195112e08ffb3f1e0277fbf9d0f35f6da7869281570e51a3b3dcbc819fef8d', 'sleeping_machines/token_readout.py': '504d4befce3e7cc70d44dd312749f0866069f0e380aa512ffb15c3e0780b5af4', 'sleeping_machines/frequency_token_readout.py': 'b67c71def15f5a4675aeb03a9d9c7d715783b51a8c661c3fbb2a4dedd7ba3b45', 'sleeping_machines/paired_route_credit.py': '23aa9cf31fcff662daab836a0746ccd4fc6f6b5bec5f9186232488d97976b98c', 'sleeping_machines/sampled_suffix_utility.py': 'e34cd6df90898c5bb2142029e6bbf6c8c6a73c7fc6b58d60da5dc62f093599d5', 'sleeping_machines/event_credit_sites.py': '3f896a7643d10ec98a75ceaa10c7217070b9e96946f58e03bac46bd45182b523', 'sleeping_machines/capacity_token_readout.py': 'fa701587411e4642a570ad0794b846f422498cd8e4ac37e100141da5bb25dd50', 'sleeping_machines/recruit_layer.py': '261e547693ce6dabf562ef113bae1a463e9118f85e9d18c06554c690586ccca4', 'sleeping_machines/compiled_episodes.py': 'b87d1e0908229da50e24168e4792e5833c2ed600beda2718ca125421ad2b8d84', 'sleeping_machines/fast_native_core.py': '33ed9a6a16d711575aaeac239d575c6c27d4b85dd9bb10d31da4d661b920386a', 'sleeping_machines/addressed_event_heads.py': 'aa43bad6c8e79e7a4e05752a96d183f05d6e40e22f5a8d2cd97e203056ca76f3', 'sleeping_machines/parallel_stream_language.py': '7237331f34b4b7802fe98896511a301480a17ebb647297dd47df94246d6ae5ce', 'sleeping_machines/sparse_race_language.py': 'ca0b90e64fd8b3395cb91f257cb743cd778da0bda86f1f2b66835b2ad970438a', 'experiments/horizon_token_language.py': '74188a3698d0240639568443f806e2dcc9b07d5c22f0179eba4a922793d349eb', 'experiments/streamed_token_stage_utility.py': 'efb008a318936923be999b8512dbc2b57e913b2f4a9efa1905addb6465208c04', 'experiments/queue/curie_data_growth_tokens_256k_b64_c16_p16_s6_20261005_v1.txt': '5b0f51dfd2508891027a1319db700e61c6fb28e2b8ed0beaf8b9940c08640100', 'experiments/queue/curie_data_growth_tokens_1m_b64_c16_p16_s6_20261006_v1.txt': '4c91522c89e60f8dffa741d5e2ca92222c52882e30a4ac87259b7e5d21bd2870'}
for source,digest in pins.items():assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest,source
for f,a in ((64,16),(16,64)):
 r=json.loads(Path(f'experiments/results/diagnostics/curie_split_horizon_64k_f{f}_a{a}_work_20261006_v1.json').read_text())
 assert r['status']=='completed' and r['optimizer_updates']==256 and r['fitting_targets']==131056
 assert r['full_numerical_state_parity'] and r['curve_parity_max_error']==0
 assert all(r[k]['formula_coverage_complete'] for k in ('fitting','inference'))
CHECK
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=2100 bash experiments/queue/run_safe.sh experiments/queue/curie_data_growth_tokens_256k_b64_c16_p16_s6_20261005_v1.txt
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_data_growth_p16_256k_streamed_utility_20261006_v1.txt
python3 - <<'CHECK'
import json
from pathlib import Path
r=json.loads(Path('experiments/results/diagnostics/curie_data_growth_p16_256k_streamed_utility_20261006_v1.json').read_text())
assert r['status']=='completed' and len(r['rows'])==1
r=r['rows'][0]
assert r['matched_rng'] and r['partition_parity'] and r['context_gain']>0
assert r['promotion']['selected_step']>0 and r['promotion']['small_fit_promotable']
CHECK
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=9000 bash experiments/queue/run_safe.sh experiments/queue/curie_data_growth_tokens_1m_b64_c16_p16_s6_20261006_v1.txt
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=900 bash experiments/queue/run_safe.sh experiments/queue/curie_data_growth_p16_1m_streamed_utility_20261006_v1.txt
