import torch, numpy as np, time
from yolo_kalman.tracker import MultiStreamTracker, TrackerConfig
S=8
t = MultiStreamTracker(S, 30.0, TrackerConfig(), device=0, emb_dim=512)
for name,g in t.graphs.items():
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(100): g.replay()
    e1.record(); torch.cuda.synchronize()
    print(name, "GPU ms/replay", e0.elapsed_time(e1)/100)
# kernel count
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    t._pre(); t._post(True); torch.cuda.synchronize()
ev=[e for e in p.events() if e.device_type.name=="CUDA"]
print("kernels pre+post_oru eager:", len(ev))
