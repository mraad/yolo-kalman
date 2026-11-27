#!/usr/bin/env bash
set -euxo pipefail
cd "$(dirname "$0")/.."
./run.sh scripts/bench.py --streams 1 4 8 16 32 --gpus 0 1 --frames 1200 --repeats 3 --assign gpu_hungarian --tag audit_gpu_hungarian
./run.sh scripts/bench.py --streams 1 4 8 16 32 --gpus 0 1 --frames 1200 --repeats 3 --assign lapjv --tag audit_lapjv
./run.sh scripts/eval_synth.py
echo AUDIT_RUN_DONE
