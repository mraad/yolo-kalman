import sys, time, numpy as np
from yolo_kalman.simrun import run_scenes
from yolo_kalman.synthetic import gap_scene, crossing_scene
from yolo_kalman.tracker import TrackerConfig
from yolo_kalman.metrics import clear_id
scenes = [gap_scene(g) for g in [10, 30, 60, 120]] + [crossing_scene()]
for ue in (True, False):
    res = {}
    for a in ("lapjv", "gpu_hungarian"):
        t = time.time()
        gts, outs, trk = run_scenes(scenes, cfg=TrackerConfig(compile=False, assign=a), use_emb=ue)
        res[a] = outs
        print(a, ue, f"{time.time()-t:.1f}s", trk.stats, [clear_id(g, o)["num_switches"] for g, o in zip(gts, outs)], trk.timing)
    diff = 0
    for oa, ob in zip(res["lapjv"], res["gpu_hungarian"]):
        for (ia, ba), (ib, bb) in zip(oa, ob):
            if not (np.array_equal(ia, ib) and np.allclose(ba, bb, atol=1e-3)): diff += 1
    print("frames differing", diff)
