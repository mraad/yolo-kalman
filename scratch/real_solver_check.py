"""Every GPU Hungarian call during a real-clip replay vs lapjv on the same matrix: total cost, assignment."""
import sys, glob, numpy as np, torch, lap
exec(open("scratch/real_replay.py").read().split("fps = [s.fps for s in p.srcs]")[0])
fps = [s.fps for s in p.srcs]
import yolo_kalman.tracker as T
from yolo_kalman.assign import lap_batch as _lb, extended_cost
st = {"problems": 0, "nonempty": 0, "cost_mismatch": 0, "assign_diff": 0, "max_abs_cost_diff": 0.0}
def checked(c, r, k, l):
    out = _lb(c, r, k, l)
    C, R, K, O = c.cpu().numpy().astype(np.float64), r.cpu().numpy(), k.cpu().numpy(), out.cpu().numpy()
    for b in range(C.shape[0]):
        st["problems"] += 1
        rows, cols = np.nonzero(R[b])[0], np.nonzero(K[b])[0]
        if len(rows) == 0 or len(cols) == 0: continue
        st["nonempty"] += 1
        sub = C[b][np.ix_(rows, cols)]
        _, x, _ = lap.lapjv(sub, extend_cost=True, cost_limit=l)
        rr = np.nonzero(x >= 0)[0]
        cl = extended_cost(sub, rr, x[rr], l)
        mr = np.nonzero(O[b] >= 0)[0]
        cg = extended_cost(sub, np.searchsorted(rows, mr), np.searchsorted(cols, O[b][mr]), l)
        d = abs(cl - cg); st["max_abs_cost_diff"] = max(st["max_abs_cost_diff"], d)
        st["cost_mismatch"] += d > 1e-9
        ga = dict(zip(np.searchsorted(rows, mr).tolist(), np.searchsorted(cols, O[b][mr]).tolist()))
        st["assign_diff"] += ga != dict(zip(rr.tolist(), x[rr].tolist()))
    return out
T.lap_batch = checked
trk = MultiStreamTracker(len(clips), fps, TrackerConfig(assign="gpu_hungarian", compile=False, cuda_graph=False), device=0, emb_dim=512)
for b, s, c, v, e, act in rec:
    trk.step(b.cuda(), s.cuda(), c.cuda(), v.cuda(), e.float().cuda(), active=act)
print("SOLVER CHECK", st)
