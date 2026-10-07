#!/usr/bin/env bash
# Third-party reproduction of the EasyTPP Taxi result (race-of-clocks model, frozen driver race_tpp_v5.py).
# Usage, from anywhere inside a clone:  bash experiments/tpp/reproduce_taxi/reproduce.sh [seeds...]   (default: 0 1 2 3 4)
# Needs Python >= 3.10 with torch (CPU build is enough) and numpy. One CPU thread, < 1 GB RAM, ~2-3 min per seed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
PY=${PYTHON:-python3}
SEEDS=${*:-0 1 2 3 4}
DRIVER=experiments/tpp/race_tpp_v5.py
DRIVER_SHA=73d2f95e11c2ebc78b1ee95340ab34584eac8d48d434353921a0cd82503b39a1
echo "$DRIVER_SHA  $DRIVER" | sha256sum -c -
mkdir -p data/easytpp/taxi
while read -r sha split; do
  f=data/easytpp/taxi/$split.json
  [ -s "$f" ] || curl -fsSL -o "$f" "https://huggingface.co/datasets/easytpp/taxi/resolve/main/$split.json"
  echo "$sha  $f" | sha256sum -c -
done <<'SUMS'
0c54b53482e792f8cd813195e76e08466a68761b35c34cd503a012b41525d989 train
9ae91594723cf9cba7333efa18f85d309f5ee76336cc25d5811d0364f05a3ddf dev
dc1e765b6626c2a231ab026606e29864c082807354066e0307494fb738bcc059 test
SUMS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STAMP=${REPRO_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
TAGS=()
for s in $SEEDS; do
  tag=thirdparty_taxi_v5_s${s}_$STAMP; TAGS+=("$tag")
  "$PY" "$DRIVER" --dataset taxi --seed "$s" --n-lognormal 8 --dropout 0.3 --patience 30 \
    --floor-cell 0.000277777777777778 --score-test --threads 1 --tag "$tag"
done
"$PY" experiments/tpp/reproduce_taxi/summarize.py "${TAGS[@]}"
