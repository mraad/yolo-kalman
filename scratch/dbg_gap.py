import sys, numpy as np
from yolo_kalman.simrun import run_scenes
from yolo_kalman.synthetic import gap_scene
from yolo_kalman.metrics import id_consistency
use_emb = sys.argv[1] == "1"
for g in [10,30,60,120]:
    sc = gap_scene(g)
    gts, outs, trk = run_scenes([sc], use_emb=use_emb)
    print(g, id_consistency(gts[0], outs[0]), trk.stats)
    # print when each id first appears
    first = {}
    for t,(ids,b) in enumerate(outs[0]):
        for i,bb in zip(ids,b):
            first.setdefault(int(i),(t,bb.round(0)))
    print("  first:", first)
    dets = sc.detections()
    # gap per object: frames with no det
    for oid in [1,2,3]:
        fr = [t for t,d in enumerate(dets) if oid in d[4]]
        miss = np.diff(fr).max()
        print("  obj",oid,"max det gap",miss)
