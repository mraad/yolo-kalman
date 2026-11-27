"""Record tracker inputs from one pipeline run over the real clips, replay through both solvers."""
import sys, glob, numpy as np, torch
from yolo_kalman.pipeline import Pipeline, PipelineConfig
from yolo_kalman.tracker import MultiStreamTracker, TrackerConfig
F = int(sys.argv[1])
clips = sorted(glob.glob("clips/*.webm"))
rec = []
cfg = PipelineConfig(max_frames=F, overlap=False, classes=[0, 1, 2, 3, 5, 7, 14, 15, 16],
                     tracker=TrackerConfig(assign="lapjv", compile=False, cuda_graph=False))
p = Pipeline(clips, gpu=0, cfg=cfg)
orig = p.tracker.step
def step(boxes, scores, cls, valid, emb=None, active=None, **kw):
    rec.append([x.cpu() for x in (boxes, scores, cls, valid)] + [emb.half().cpu(), np.array(active)])
    return orig(boxes, scores, cls, valid, emb, active=active, **kw)
p.tracker.step = step
p.run()
fps = [s.fps for s in p.srcs]
print("recorded", len(rec), "steps")
for comp in (False, True):
    outs = {}
    for a in ("lapjv", "gpu_hungarian"):
        trk = MultiStreamTracker(len(clips), fps, TrackerConfig(assign=a, compile=comp, cuda_graph=comp), device=0, emb_dim=512)
        o = []
        for b, s, c, v, e, act in rec:
            o.append(trk.step(b.cuda(), s.cuda(), c.cuda(), v.cuda(), e.float().cuda(), active=act))
        outs[a] = (o, dict(trk.stats))
    (oa, sa), (ob, sb) = outs["lapjv"], outs["gpu_hungarian"]
    ndiff_id = ndiff_box = 0; first = None
    ids_a, ids_b = set(), set()
    for t, (xa, xb) in enumerate(zip(oa, ob)):
        for s, (ya, yb) in enumerate(zip(xa, xb)):
            if ya is None: continue
            ids_a |= {(s, int(i)) for i in ya.ids}; ids_b |= {(s, int(i)) for i in yb.ids}
            if not np.array_equal(ya.ids, yb.ids):
                ndiff_id += 1; first = first if first is not None else (t, s)
            elif not np.allclose(ya.boxes, yb.boxes, atol=1e-3):
                ndiff_box += 1
    print(f"compiled={comp}: stream-frames with different ids {ndiff_id}, with box diff >1e-3 px {ndiff_box}, "
          f"first id diff {first}; unique ids lapjv {len(ids_a)} gpu {len(ids_b)}")
    print("   stats lapjv", sa); print("   stats gpu  ", sb)
# strict-limit experiment: forbid c == limit in the GPU solver (eager)
import yolo_kalman.tracker as T
from yolo_kalman.assign import lap_batch as _lb
T.lap_batch = lambda c, r, k, l: _lb(c, r, k, float(np.nextafter(l, -np.inf)))
trk = MultiStreamTracker(len(clips), fps, TrackerConfig(assign="gpu_hungarian", compile=False, cuda_graph=False), device=0, emb_dim=512)
ob = [trk.step(b.cuda(), s.cuda(), c.cuda(), v.cuda(), e.float().cuda(), active=act) for b, s, c, v, e, act in rec]
oa = outs["lapjv"][0]
nid = nbox = 0
for xa, xb in zip(oa, ob):
    for ya, yb in zip(xa, xb):
        if ya is None: continue
        if not np.array_equal(ya.ids, yb.ids): nid += 1
        elif not np.allclose(ya.boxes, yb.boxes, atol=1e-3): nbox += 1
print("strict c<limit: id diffs", nid, "box diffs", nbox, trk.stats)
