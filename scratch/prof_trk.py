import sys, time, numpy as np, torch
from yolo_kalman.synthetic import gap_scene, crossing_scene
from yolo_kalman.tracker import MultiStreamTracker, TrackerConfig
S = int(sys.argv[1]) if len(sys.argv) > 1 else 8
assign = sys.argv[2] if len(sys.argv) > 2 else "gpu_hungarian"
scenes = [gap_scene([10, 30, 60, 120][i % 4]) for i in range(S)]
# more objects per scene: tile scenes with clutter
dets = [sc.detections(seed=i, emb_dim=512, fp_rate=3.0) for i, sc in enumerate(scenes)]
T = min(sc.length for sc in scenes)
trk = MultiStreamTracker(S, 30.0, TrackerConfig(assign=assign), device=0, emb_dim=512)
N = 160
batches = []
for t in range(T):
    B = np.zeros((S, N, 4), np.float32); Sc = np.zeros((S, N), np.float32); C = np.zeros((S, N), np.int64)
    V = np.zeros((S, N), bool); E = np.zeros((S, N, 512), np.float32)
    for s in range(S):
        b, sc_, c, e, _ = dets[s][t]; n = len(sc_)
        B[s, :n], Sc[s, :n], C[s, :n], V[s, :n], E[s, :n] = b, sc_, c, True, e
    batches.append([torch.from_numpy(x).cuda() for x in (B, Sc, C, V, E)])
print("avg dets/stream", np.mean([b[3].sum().item() / S for b in batches]))
def run(nsteps):
    for t in range(nsteps):
        b = batches[t % T]
        if assign == "gpu_hungarian":
            trk.submit(*b)
            if (t + 1) % 8 == 0: trk.collect()
        else:
            sc_np = b[1].cpu().numpy(); trk.step(*b)
    if assign == "gpu_hungarian": trk.collect()
run(16); torch.cuda.synchronize()
for k in trk.timing: trk.timing[k] = 0.0
t0 = time.perf_counter(); n = 400; run(n); torch.cuda.synchronize(); w = time.perf_counter() - t0
print(assign, S, f"wall/step {w/n*1e3:.3f} ms", {k: round(v / n * 1e3, 3) for k, v in trk.timing.items()}, trk.stats)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    run(8); torch.cuda.synchronize()
ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
print("cuda kernels per step", len(ev) / 8)
import collections
cnt = collections.Counter(e.name[:90] for e in ev)
tot = collections.defaultdict(float)
for e in ev: tot[e.name[:90]] += e.device_time
for k, v in sorted(tot.items(), key=lambda x: -x[1])[:25]: print(f"{v/8:8.1f}us {cnt[k]/8:5.1f}x {k}")
