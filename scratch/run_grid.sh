#!/bin/bash
# full benchmark grid, both solvers; each run under a timeout, retried once (cold-compile hang guard)
cd "$(dirname "$0")/.."
for a in gpu_hungarian lapjv; do
  for s in 1 4 8 16 32; do
    for try in 1 2 3; do
      timeout 600 ./run.sh scripts/bench.py --streams $s --gpus 0 1 --frames 1200 --tag final_${a}_s$s --assign $a \
        > scratch/final_${a}_s${s}_t$try.log 2>&1
      rc=$?
      grep -E "^\[final|Error" scratch/final_${a}_s${s}_t$try.log
      echo "assign=$a streams=$s try=$try rc=$rc"
      [ $rc -eq 0 ] && break
      pkill -f "spawn_main" ; sleep 3
    done
  done
done
echo GRID_DONE
