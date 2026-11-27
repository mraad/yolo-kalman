#!/bin/bash
# usage: watch_probe.sh LOG  -- prints a status line every 60 s until EXIT appears (max 9 min)
LOG=$1
for i in 1 2 3 4 5 6 7 8 9; do
  sleep 60
  st=$(grep -E '^\[|EXIT|Error' "$LOG" | tail -2 | tr '\n' ' ')
  pr=$(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{printf "%s:%s%% ", $2, $3}')
  gp=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')
  echo "t=$i min: $st procs=$pr gpu=$gp"
  grep -q EXIT "$LOG" && break
done
