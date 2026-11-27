"""Pre-compile tracker graphs for every per-worker stream count (sequential, one process)."""
import sys, time, torch
from yolo_kalman.tracker import MultiStreamTracker, TrackerConfig
for a in sys.argv[1].split(","):
    for S in [int(x) for x in sys.argv[2].split(",")]:
        t = time.time()
        MultiStreamTracker(S, 30.0, TrackerConfig(assign=a), device=1, emb_dim=512)
        print(a, S, f"{time.time() - t:.0f}s", flush=True)
