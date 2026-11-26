import numpy as np, torch
from yolo_kalman.pipeline import Pipeline, PipelineConfig
from yolo_kalman.tracker import TrackerConfig, LOST
cfg = PipelineConfig(use_reid=False, classes=[0], overlap=False, tracker=TrackerConfig(compile=False, cuda_graph=False))
p = Pipeline(["data/synth/synth_gap030.mp4"], cfg=cfg)
trk = p.tracker
orig = trk.step
log = []
def step(*a, **k):
    out = orig(*a, **k)
    f = trk.frame[0]
    sl = np.nonzero(trk.ids[0] == 1)[0]
    if 86 <= f <= 108 and len(sl):
        s = sl[0]
        host = trk.cost_host.numpy()[0]
        iou, d2 = host[0, s], host[1, s]
        sc = k['scores_np'][0]
        j = np.nonzero(iou > 0.01)[0]
        print(f, "good", trk.post_host.numpy()[2,0,sl[0]], "ref", trk.ref_wh[0,sl[0]].round(0), "state", trk.state[0, s], "tsu", trk.tsu[0, s], "mean", trk.mean[0, s, :4].cpu().numpy().round(0),
              "v", trk.mean[0, s, 4:6].cpu().numpy().round(2),
              "dets(iou,d2,score,w)", [(round(float(iou[x]),2), round(float(d2[x]),1), round(float(sc[x]),2), round(float(trk.det_host.numpy()[0, x, 1]),0)) for x in j],
              "new ids", [int(i) for i in out[0].ids if i > 3])
    return out
trk.step = step
p.run()
