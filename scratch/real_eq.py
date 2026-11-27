"""Compare gpu_hungarian vs lapjv outputs on real clips (eager or compiled), first divergence frame."""
import sys, glob, numpy as np, torch
from yolo_kalman.pipeline import Pipeline, PipelineConfig
from yolo_kalman.tracker import TrackerConfig
comp = sys.argv[1] == "compiled"
clips = sorted(glob.glob("clips/*.webm"))
res = {}
A = sys.argv[3].split(",") if len(sys.argv) > 3 else ["lapjv", "gpu_hungarian"]
for k, a in enumerate(A):
    cfg = PipelineConfig(max_frames=int(sys.argv[2]), overlap=len(sys.argv) <= 4, classes=[0, 1, 2, 3, 5, 7, 14, 15, 16],
                         tracker=TrackerConfig(assign=a, compile=comp, cuda_graph=comp))
    p = Pipeline(clips, gpu=0, cfg=cfg)
    res[k] = p.run()
for ra, rb in zip(res[0], res[1]):
    first = None
    for t, (x, y) in enumerate(zip(ra.tracks, rb.tracks)):
        same = np.array_equal(x[0], y[0]) and x[1].shape == y[1].shape and np.allclose(x[1], y[1], atol=1e-2)
        if not same:
            first = t; break
    print(ra.path.split("/")[-1][:40], "frames", len(ra.tracks), "first divergence", first)
    if first is not None:
        x, y = ra.tracks[first], rb.tracks[first]
        print("   lapjv ids", x[0][:12], " gpu ids", y[0][:12])
        if x[1].shape == y[1].shape: print("   max box diff", np.abs(x[1] - y[1]).max())
