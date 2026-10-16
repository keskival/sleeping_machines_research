#!/usr/bin/env bash
# Bootstrap a disposable single-GPU worker (Hetzner GEX131 / GEX63, Ubuntu 24.04) for the R1 GPU scaling ladder.
# Runbook: experiments/runbooks/GPU_SCALING_LADDER.md. Run as root after first login; idempotent.
#   SHARDS=60 TORCH_VERSION=2.14.1 TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/gpu_worker_setup.sh
set -euo pipefail
SHARDS=${SHARDS:-60}                       # 100M tokens each; 60 covers the largest planned arm (~5.1B tokens) with margin
TORCH_VERSION=${TORCH_VERSION:-2.14.1}     # the project venv's version; must ship sm_120 (Blackwell) kernels
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

# 1. NVIDIA driver (Blackwell needs the open kernel modules, driver >= 570) and basics
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  apt-get update && apt-get install -y ubuntu-drivers-common
  ubuntu-drivers install --gpgpu || apt-get install -y nvidia-driver-570-server-open
  echo "Driver installed: reboot, then rerun this script." && exit 0
fi
apt-get install -y git tmux python3-venv python3-dev build-essential htop nvtop rsync
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv

# 2. Repository at /workspace (run_safe.sh expects /workspace and /workspace/.venv-docker)
if [ ! -d /workspace/.git ]; then
  echo "Clone the repository first: add the worker's deploy key (read/write) to GitHub, then"
  echo "  git clone git@github.com:keskival/sleeping_machines_research.git /workspace" && exit 1
fi
cd /workspace
[ -x .venv-docker/bin/python ] || python3 -m venv .venv-docker
.venv-docker/bin/pip install --upgrade pip
.venv-docker/bin/pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"
.venv-docker/bin/pip install numpy scipy matplotlib huggingface_hub
.venv-docker/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); torch.cuda.synchronize()
import time; t = time.time()
for _ in range(50): y = x @ x
torch.cuda.synchronize(); print('bf16 matmul TFLOP/s', 50*2*4096**3/(time.time() - t)/1e12)
EOF

# 3. Data: FineWeb GPT-2 shards (llm.c format, kjj0/fineweb10B-gpt2) and the GPT-2 vocabulary for token roles
mkdir -p data/fineweb_gpt2 data/gpt2
.venv-docker/bin/python - "$SHARDS" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
n = int(sys.argv[1])
for name in ['fineweb_val_000000.bin'] + [f'fineweb_train_{i:06d}.bin' for i in range(1, n + 1)]:
    hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename=name, repo_type='dataset', local_dir='data/fineweb_gpt2')
print('shards ready', n)
EOF
[ -f data/gpt2/encoder.json ] || curl -fsSL -o data/gpt2/encoder.json https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json
# Identity against the Curie copies the CPU ladder used
sha256sum -c - <<'EOF'
771fa4a99b9fe0946ffb6e848b4ba5c6a9b0fe87860ebf03bc2c1c7e45f8178e  data/fineweb_gpt2/fineweb_train_000001.bin
5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640  data/fineweb_gpt2/fineweb_val_000000.bin
196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783  data/gpt2/encoder.json
EOF

# 4. Contract on this machine (CPU fp32 kernels vs DeepKeyedLM), then a CUDA smoke
.venv-docker/bin/python experiments/r1_gpu_ladder.py --mode contract | tail -3
echo "Setup complete. Next: runbook step 4 (GPU equivalence and smoke)."
