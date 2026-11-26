"""Run the tracker on synthetic scenes (simulated detections), one scene per stream, batched."""
from __future__ import annotations

import numpy as np
import torch

from .tracker import MultiStreamTracker, TrackerConfig


def run_scenes(scenes, cfg: TrackerConfig | None = None, use_emb=True, seed=0, device=0, emb_dim=64,
               min_vis=0.3, **detkw):
    """GT is restricted to objects with visibility >= min_vis (fully hidden objects are not scored)."""
    S = len(scenes)
    dets = [sc.detections(seed=seed + i, emb_dim=emb_dim, **detkw) for i, sc in enumerate(scenes)]
    T = max(sc.length for sc in scenes)
    trk = MultiStreamTracker(S, [sc.fps for sc in scenes], cfg, device=device, emb_dim=emb_dim)
    dev = trk.dev
    outs = [[] for _ in range(S)]
    for t in range(T):
        active = np.array([t < sc.length for sc in scenes])
        N = max(1, max(len(dets[s][t][1]) for s in range(S) if active[s]) if active.any() else 1)
        B = np.zeros((S, N, 4), np.float32); Sc = np.zeros((S, N), np.float32)
        C = np.zeros((S, N), np.int64); V = np.zeros((S, N), bool); E = np.zeros((S, N, emb_dim), np.float32)
        for s in range(S):
            if not active[s]:
                continue
            b, sc_, c, e, _ = dets[s][t]
            n = len(sc_)
            B[s, :n], Sc[s, :n], C[s, :n], V[s, :n], E[s, :n] = b, sc_, c, True, e
        tt = lambda x: torch.from_numpy(x).to(dev)
        res = trk.step(tt(B), tt(Sc), tt(C), tt(V), tt(E) if use_emb else None, active=active)
        for s in range(S):
            if res[s] is not None:
                outs[s].append((res[s].ids, res[s].boxes.cpu().numpy()))
    gts = []
    for sc in scenes:
        g = []
        for fr in sc.gt():
            fr = [x for x in fr if x[2] >= min_vis]
            g.append((np.array([x[0] for x in fr], np.int64), np.array([x[1] for x in fr]).reshape(-1, 4)))
        gts.append(g)
    return gts, outs, trk
