import time, numpy as np, torch, lap
from yolo_kalman.assign import lap_batch
rng = np.random.default_rng(0)
def tracking_like(B, n, m, noise=0.15):
    c = np.full((B, n, m), 6e4, np.float32)
    for b in range(B):
        perm = rng.permutation(m)
        for i in range(n):
            # true det plus a few neighbours with plausible cost
            c[b, i, perm[i % m]] = rng.random() * 0.3
            for k in rng.choice(m, 3): c[b, i, k] = min(c[b, i, k], 0.4 + rng.random() * 0.5)
    return c
for B, n, m in [(8, 20, 20), (16, 30, 30), (16, 60, 60), (32, 40, 40), (32, 150, 150)]:
    c = tracking_like(B, n, m)
    tc = torch.from_numpy(c).cuda(); rm = torch.ones(B, n, dtype=bool, device='cuda'); cm = torch.ones(B, m, dtype=bool, device='cuda')
    lap_batch(tc, rm, cm, 0.8); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(50): lap_batch(tc, rm, cm, 0.8)
    torch.cuda.synchronize(); g = (time.perf_counter()-t)/50*1e3
    cd = c.astype(np.float64); t = time.perf_counter()
    for _ in range(10):
        for b in range(B): lap.lapjv(cd[b], extend_cost=True, cost_limit=0.8)
    print(B, n, m, f"gpu {g:.3f} ms  lapjv {(time.perf_counter()-t)/10*1e3:.3f} ms")
