import time, numpy as np, torch, lap
from scipy.optimize import linear_sum_assignment
from yolo_kalman.assign import lap_batch, extended_cost

def ref(c, limit):
    n, m = c.shape
    if n == 0 or m == 0: return limit/2*(n+m)
    e = np.full((n+m, n+m), limit/2); e[n:, m:] = 0; e[:n, :m] = c
    r, k = linear_sum_assignment(e); return e[r, k].sum()

rng = np.random.default_rng(0)
t0=time.time(); torch.zeros(1,device='cuda'); lap_batch(torch.zeros(1,2,2,device='cuda'), torch.ones(1,2,dtype=bool,device='cuda'), torch.ones(1,2,dtype=bool,device='cuda'), 0.5); print('build', time.time()-t0)
bad=0
for trial in range(0):
    B, R, C = 8, int(rng.integers(1, 40)), int(rng.integers(1, 40))
    cost = rng.random((B, R, C)).astype(np.float32)
    if trial % 3 == 0: cost = np.round(cost * 4) / 4   # ties
    cost[rng.random(cost.shape) < 0.5] = 6e4
    rm = rng.random((B, R)) < 0.8; cm = rng.random((B, C)) < 0.8
    limit = float(rng.choice([0.25, 0.5, 0.8]))
    tc = torch.from_numpy(cost).cuda(); trm = torch.from_numpy(rm).cuda(); tcm = torch.from_numpy(cm).cuda()
    for fn in (lap_batch,):
        out = fn(tc, trm, tcm, limit).cpu().numpy()
        for b in range(B):
            rows, cols = np.nonzero(rm[b])[0], np.nonzero(cm[b])[0]
            sub = cost[b][np.ix_(rows, cols)].astype(np.float64)
            o = out[b]
            assert (o[~rm[b]] == -1).all()
            mr = np.nonzero(o >= 0)[0]
            assert rm[b][mr].all() and cm[b][o[mr]].all() and len(set(o[mr])) == len(mr)
            rr = np.searchsorted(rows, mr); cc = np.searchsorted(cols, o[mr])
            got = extended_cost(sub, rr, cc, limit)
            want = ref(sub, limit)
            if sub.size:
                _, x, _ = lap.lapjv(sub, extend_cost=True, cost_limit=limit)
                r2 = np.nonzero(x >= 0)[0]; lj = extended_cost(sub, r2, x[r2], limit)
            else: lj = want
            if abs(got-want) > 1e-6 or abs(lj-want) > 1e-6:
                bad += 1; print(getattr(fn, "__name__", "cuda"), trial, b, got, want, lj)
print('bad', bad)
# timing
for B, n in [(8, 20), (16, 40), (32, 160)]:
    cost = torch.rand(B, n, n, device='cuda'); rm = torch.ones(B, n, dtype=bool, device='cuda'); cm = rm.clone()
    for fn in (lap_batch,):
        fn(cost, rm, cm, 0.8); torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(10): fn(cost, rm, cm, 0.8)
        torch.cuda.synchronize(); print(getattr(fn, "__name__", "lap_batch_cuda"), B, n, (time.perf_counter()-t)/10*1e3, 'ms')
    c = cost[0].cpu().numpy().astype(np.float64); t=time.perf_counter()
    for _ in range(10):
        for b in range(B): lap.lapjv(c, extend_cost=True, cost_limit=0.8)
    print('lapjv cpu', B, n, (time.perf_counter()-t)/10*1e3, 'ms')
