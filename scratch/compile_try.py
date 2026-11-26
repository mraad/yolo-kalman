import torch, time, sys
from yolo_kalman.tracker import MultiStreamTracker, TrackerConfig
S=16
for comp in (False, True):
    t0=time.time()
    t = MultiStreamTracker(S, 30.0, TrackerConfig(compile=comp), device=0, emb_dim=512)
    print("compile", comp, "init s", round(time.time()-t0,1))
    for name,g in t.graphs.items():
        for _ in range(3): g.replay()
        torch.cuda.synchronize()
        e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(100): g.replay()
        e1.record(); torch.cuda.synchronize()
        print("  ", name, "GPU ms/replay", round(e0.elapsed_time(e1)/100,3))
