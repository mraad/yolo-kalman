import numpy as np, torch
from yolo_kalman import tracker as T
from yolo_kalman.synthetic import gap_scene
from yolo_kalman import kalman as kf
sc = gap_scene(10); dets = sc.detections(emb_dim=64)
trk = T.MultiStreamTracker(1, 30.0, device=0, emb_dim=64)
for t in range(130):
    b,s,c,e,oid = dets[t]; n=len(s)
    tt=lambda x: torch.from_numpy(np.asarray(x)).cuda()[None]
    if n==0: b=np.zeros((1,4),np.float32); s=np.zeros(1,np.float32); c=np.zeros(1,np.int64); v=np.zeros(1,bool)
    else: v=np.ones(n,bool)
    out = trk.step(tt(b),tt(s),tt(c),tt(v),None)
    sl = np.nonzero(trk.ids[0]==2)[0]
    if 60<t<110 and len(sl):
        sl=sl[0]
        print(t, "state",trk.state[0,sl],"tsu",trk.tsu[0,sl],"mean",trk.mean[0,sl].cpu().numpy().round(1), "cov_x", trk.cov[0,sl,0,0].item().__round__(1), "dets", [(o, bb.round(0).tolist(), round(float(ss),2)) for o,bb,ss in zip(oid,b,s) if o==2])
