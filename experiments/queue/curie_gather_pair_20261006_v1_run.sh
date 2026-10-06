#!/usr/bin/env bash
set -euo pipefail
cd /workspace
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_control_tokens_8k_b64_c16_p24_s6_20261006_v1.txt
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_gather_tokens_8k_b64_c16_p24_s6_20261006_v1.txt
