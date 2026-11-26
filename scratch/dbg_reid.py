import numpy as np
from yolo_kalman.pipeline import Pipeline, PipelineConfig
from yolo_kalman.tracker import TrackerConfig, LOST
cfg = PipelineConfig(classes=[0], overlap=False, tracker=TrackerConfig(compile=False))
p = Pipeline(["data/synth/synth_gap120.mp4"], cfg=cfg)
trk = p.tracker; orig = trk.step
def step(*a, **k):
    out = orig(*a, **k)
    f = trk.frame[0]
    s3 = np.nonzero(trk.ids[0] == 3)[0]; s4 = np.nonzero(trk.ids[0] == 4)[0]
    if len(s3) and len(s4) and f % 2 == 0:
        a_, b_ = s3[0], s4[0]
        print(f, "st3", trk.state[0,a_], "tsu3", trk.tsu[0,a_], "age4", f - trk.birth[0,b_], "emb_dist", trk.tt_host.numpy()[0,0,a_,b_], "d2", trk.tt_host.numpy()[0,1,a_,b_],
              "pos3", trk.mean[0,a_,:4].cpu().numpy().round(0), "pos4", trk.mean[0,b_,:4].cpu().numpy().round(0), "nemb3", trk.emb_count[0,a_], "nemb4", trk.emb_count[0,b_])
    return out
trk.step = step
p.run()
print(trk.stats)
