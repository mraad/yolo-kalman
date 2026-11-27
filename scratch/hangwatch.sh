#!/bin/bash
# If a grid run log has not changed for 240 s while bench workers exist, dump their stacks (SIGUSR1) once.
cd "$(dirname "$0")"
while ! grep -q GRID_DONE grid.log; do
  sleep 30
  f=$(ls -t grid_*.log | head -1)
  age=$(( $(date +%s) - $(stat -c %Y "$f") ))
  if [ $age -gt 240 ] && ! grep -q "^\[hung" "$f"; then
    for p in $(pgrep -f "multiprocessing.spawn import spawn_main"); do kill -USR1 $p; done
    echo "dumped stacks into $f at age $age"; sleep 400
  fi
done
