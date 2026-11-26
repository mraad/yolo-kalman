"""Tracking metrics: HOTA (re-implementation of the TrackEval algorithm), CLEAR/ID via motmetrics."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    aa = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)
    ab = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(-1)
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def hota(gt_frames, tr_frames):
    """gt_frames / tr_frames: list over frames of (ids [n], boxes [n,4] xyxy).

    Returns dict(HOTA, DetA, AssA, LocA) averaged over alpha = 0.05..0.95 (TrackEval definition).
    """
    alphas = np.arange(0.05, 0.99, 0.05)
    gt_ids = sorted({int(i) for ids, _ in gt_frames for i in ids})
    tr_ids = sorted({int(i) for ids, _ in tr_frames for i in ids})
    gmap = {g: k for k, g in enumerate(gt_ids)}
    tmap = {t: k for k, t in enumerate(tr_ids)}
    G, T = len(gt_ids), len(tr_ids)
    if G == 0 or T == 0:
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0, "LocA": 0.0}
    pot = np.zeros((G, T))
    gcount = np.zeros(G)
    tcount = np.zeros(T)
    sims = []
    for (gi, gb), (ti, tb) in zip(gt_frames, tr_frames):
        gi = np.array([gmap[int(i)] for i in gi], int)
        ti = np.array([tmap[int(i)] for i in ti], int)
        S = iou_xyxy(gb, tb) if len(gi) and len(ti) else np.zeros((len(gi), len(ti)))
        sims.append((gi, ti, S))
        gcount[gi] += 1
        tcount[ti] += 1
        if len(gi) and len(ti):
            denom = S.sum(1, keepdims=True) + S.sum(0, keepdims=True) - S
            sim_iou = np.divide(S, denom, out=np.zeros_like(S), where=denom > 1e-12)
            pot[np.ix_(gi, ti)] += sim_iou
    gas = pot / (gcount[:, None] + tcount[None, :] - pot)
    A = len(alphas)
    tp = np.zeros(A)
    fn = np.zeros(A)
    fp = np.zeros(A)
    loc = np.zeros(A)
    mcount = np.zeros((A, G, T))
    for gi, ti, S in sims:
        if len(gi) == 0 or len(ti) == 0:
            fn += len(gi)
            fp += len(ti)
            continue
        score = gas[np.ix_(gi, ti)] * S
        r, c = linear_sum_assignment(-score)
        for a, al in enumerate(alphas):
            ok = S[r, c] >= al - 1e-10
            rr, cc = r[ok], c[ok]
            n = len(rr)
            tp[a] += n
            fn[a] += len(gi) - n
            fp[a] += len(ti) - n
            loc[a] += S[rr, cc].sum()
            mcount[a, gi[rr], ti[cc]] += 1
    assa = np.zeros(A)
    for a in range(A):
        mc = mcount[a]
        ass = mc / np.maximum(1, gcount[:, None] + tcount[None, :] - mc)
        assa[a] = (mc * ass).sum() / max(1, tp[a])
    deta = tp / np.maximum(1, tp + fn + fp)
    h = np.sqrt(deta * assa)
    loca = loc / np.maximum(1e-10, tp)
    return {"HOTA": float(h.mean()), "DetA": float(deta.mean()), "AssA": float(assa.mean()),
            "LocA": float(loca.mean())}


def clear_id(gt_frames, tr_frames, iou_thresh=0.5):
    """MOTA / IDF1 / ID switches / fragmentations via motmetrics."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    for (gi, gb), (ti, tb) in zip(gt_frames, tr_frames):
        d = 1 - iou_xyxy(gb, tb) if len(gi) and len(ti) else np.zeros((len(gi), len(ti)))
        d = np.where(d > 1 - iou_thresh, np.nan, d)
        acc.update(list(map(int, gi)), list(map(int, ti)), d)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["mota", "idf1", "num_switches", "num_fragmentations", "num_unique_objects",
                                 "precision", "recall"], name="x")
    return {k: float(s[k].iloc[0]) for k in s.columns}


def id_consistency(gt_frames, tr_frames, iou_thresh=0.5):
    """For each GT id: the set of track ids matched to it (per-frame Hungarian on IoU)."""
    m: dict[int, set] = {}
    for (gi, gb), (ti, tb) in zip(gt_frames, tr_frames):
        if not len(gi) or not len(ti):
            continue
        S = iou_xyxy(gb, tb)
        r, c = linear_sum_assignment(-S)
        for a, b in zip(r, c):
            if S[a, b] >= iou_thresh:
                m.setdefault(int(gi[a]), set()).add(int(ti[b]))
    return m
