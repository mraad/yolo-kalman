"""Batched multi-stream tracker: Kalman (GPU) + BYTE + OC-SORT (ORU/OCM/OCR) + appearance re-ID.

Data layout
-----------
Numeric track state lives on the GPU in fixed-capacity slot tensors ``[S, M, ...]`` (S streams,
M slots per stream).  Small integer bookkeeping (slot state, ids, ages) is mirrored on the CPU,
because the sequential matching cascade is decided there anyway.

Per frame (all streams at once):
  1. GPU: KF predict for every live slot.
  2. GPU: build *all* cost terms for the cascade in one shot, for compacted candidate tracks x
     detections: IoU(pred, det), Mahalanobis d^2, appearance cosine distance, OCM direction term,
     IoU(last observation, det).  One device->host copy.
  3. CPU: matching cascade per stream with ``lap.lapjv`` (rectangular, cost_limit):
       S1  confirmed (tracked + lost) x high dets   fused IoU/appearance + OCM, Mahalanobis gate
       S2  tracked-last-frame x low dets             IoU (BYTE second association)
       S3  remaining x remaining high dets           IoU(last observation, det) (OC-SORT OCR)
       S4  lost x remaining high dets                appearance only, loose Mahalanobis gate (re-ID)
       S5  tentative x remaining high dets           IoU
       new tentative tracks from leftover high dets
  4. GPU: KF update of matched slots (NSA noise).  Slots re-matched after a gap get the OC-SORT
     observation-centric re-update (ORU): roll back to the last observed state and re-run
     predict/update along a virtual linear trajectory to the new detection.  Appearance EMA update.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lap
import numpy as np
import torch
import torch.nn.functional as F

from . import kalman as kf

FREE, TENTATIVE, TRACKED, LOST = 0, 1, 2, 3
BIG = 1e5

# COCO class groups whose labels flicker between each other (car/bus/truck, bicycle/motorcycle)
DEFAULT_CLASS_GROUPS = {2: 2, 5: 2, 7: 2, 1: 1, 3: 1}


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5        # BYTE: first-association detections
    low_thresh: float = 0.1         # BYTE: second-association detections (low..high)
    new_track_thresh: float = 0.6   # min score to spawn a track
    match_thresh: float = 0.8       # S1 max fused cost
    low_match_thresh: float = 0.5   # S2 max 1-IoU
    ocr_thresh: float = 0.7         # S3 max 1-IoU(last obs, det)
    unconfirmed_thresh: float = 0.7  # S5 max 1-IoU
    gate_chi2: float = 13.277       # chi2(4 dof, 0.99): Mahalanobis gate for S1/S2
    reid_gate_chi2: float = 60.0    # loose gate for S4 (KF covariance grows while coasting)
    lost_buffer_s: float = 6.0      # how long lost tracks coast before deletion (seconds)
    ocr_max_gap: int = 30           # S3 only for tracks lost <= this many frames
    n_init: int = 2                 # hits to confirm a tentative track
    # appearance
    use_appearance: bool = True
    emb_alpha: float = 0.9          # EMA factor (dynamic, Deep OC-SORT style)
    emb_thresh: float = 0.25        # S1: use appearance if cos-dist below this ...
    proximity_thresh: float = 0.5   # ... and 1-IoU below this (BoT-SORT fusion)
    reid_thresh: float = 0.25       # S4: max cos-dist for re-identification
    emb_update_max_overlap: float = 0.4  # skip EMA update when det overlaps another det (occluder)
    # OC-SORT
    ocm_weight: float = 0.2
    ocm_delta: int = 3
    use_oru: bool = True
    oru_max_steps: int = 8
    freeze_size_when_lost: bool = True
    # low-quality observations (score < high_thresh, or box abruptly shrinks = partial occlusion):
    # inflate R, and do not use them as OC-SORT "observations" (snapshot / last_obs / OCM history)
    low_quality_r_scale: float = 100.0
    shape_change_ratio: float = 0.75
    class_aware: bool = True
    class_groups: dict = field(default_factory=lambda: dict(DEFAULT_CLASS_GROUPS))
    max_tracks: int = 256           # slots per stream
    min_box_area: float = 10.0


@dataclass
class FrameOutput:
    ids: np.ndarray       # [n] int64
    boxes: torch.Tensor   # [n,4] xyxy (GPU; KF posterior)
    scores: np.ndarray    # [n]
    cls: np.ndarray       # [n]


def _lap(cost: np.ndarray, limit: float):
    """Return (rows, cols) of the optimal assignment with every cost <= limit."""
    if cost.size == 0:
        return np.empty(0, int), np.empty(0, int)
    _, x, _ = lap.lapjv(cost.astype(np.float64), extend_cost=True, cost_limit=limit)
    r = np.nonzero(x >= 0)[0]
    return r, x[r]


class MultiStreamTracker:
    def __init__(self, num_streams: int, fps: float | list[float] = 30.0, cfg: TrackerConfig | None = None,
                 device: int | str = 0, emb_dim: int = 512):
        self.cfg = cfg or TrackerConfig()
        self.S, self.M, self.D = num_streams, self.cfg.max_tracks, emb_dim
        self.dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        fps = fps if isinstance(fps, (list, tuple)) else [fps] * num_streams
        self.buffer = np.array([max(1, int(round(f * self.cfg.lost_buffer_s))) for f in fps])
        S, M, D, dev = self.S, self.M, self.D, self.dev
        # GPU numeric state
        self.mean = torch.zeros(S, M, 8, device=dev)
        self.mean[..., 2:4] = 1.0
        self.cov = torch.eye(8, device=dev).repeat(S, M, 1, 1)
        self.snap_mean = self.mean.clone()          # posterior at last observation (for ORU)
        self.snap_cov = self.cov.clone()
        self.last_obs = torch.zeros(S, M, 4, device=dev)   # xywh
        self.hist = torch.zeros(S, M, self.cfg.ocm_delta + 1, 4, device=dev)  # xywh, newest first
        self.emb = torch.zeros(S, M, D, device=dev)
        # CPU bookkeeping
        self.state = np.zeros((S, M), np.int8)
        self.ids = np.full((S, M), -1, np.int64)
        self.tsu = np.zeros((S, M), np.int32)       # frames since last update
        self.tsg = np.zeros((S, M), np.int32)       # frames since last good (trusted) observation
        self.hits = np.zeros((S, M), np.int32)
        self.nobs = np.zeros((S, M), np.int32)
        self.has_emb = np.zeros((S, M), bool)
        self.cls = np.zeros((S, M), np.int64)
        self.score = np.zeros((S, M), np.float32)
        self.next_id = np.ones(S, np.int64)
        self.frame = np.zeros(S, np.int64)
        self.timing = {"gpu_cost": 0.0, "cpu_match": 0.0, "gpu_update": 0.0}
        self.stats = {"oru": 0, "reid": 0, "ocr": 0}

    # ------------------------------------------------------------------------------------------
    def reset_stream(self, s: int, fps: float | None = None):
        self.state[s] = FREE
        self.ids[s] = -1
        self.tsu[s] = self.tsg[s] = self.hits[s] = self.nobs[s] = 0
        self.has_emb[s] = False
        self.next_id[s] = 1
        self.frame[s] = 0
        if fps:
            self.buffer[s] = max(1, int(round(fps * self.cfg.lost_buffer_s)))

    def _group(self, c: np.ndarray) -> np.ndarray:
        g = self.cfg.class_groups
        if not g:
            return c
        return np.vectorize(lambda v: g.get(int(v), 1000 + int(v)), otypes=[np.int64])(c) if c.size else c

    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, boxes: torch.Tensor, scores: torch.Tensor, cls: torch.Tensor, valid: torch.Tensor,
             emb: torch.Tensor | None = None, active: np.ndarray | None = None,
             scores_np: np.ndarray | None = None, cls_np: np.ndarray | None = None,
             valid_np: np.ndarray | None = None) -> list[FrameOutput | None]:
        """One frame for every stream.

        boxes [S,N,4] xyxy (source px), scores [S,N], cls [S,N], valid [S,N] (GPU tensors),
        emb [S,N,D] unit vectors or None.  ``active[s]=False`` skips stream s (no frame).
        ``*_np`` optional host copies (saves one transfer if the caller already has them).
        """
        cfg, S, dev = self.cfg, self.S, self.dev
        active = np.ones(S, bool) if active is None else np.asarray(active, bool)
        N = boxes.shape[1]
        t0 = _now()
        live = (self.state != FREE) & active[:, None]
        self.frame[active] += 1

        # ---- 1. predict ----------------------------------------------------------------------
        live_t = torch.from_numpy(live).to(dev, non_blocking=True)
        mean = self.mean
        if cfg.freeze_size_when_lost:
            lost_t = torch.from_numpy((self.state == LOST) & live).to(dev, non_blocking=True)
            mean = mean.clone()
            mean[..., 6:8] = torch.where(lost_t[..., None], 0.0, mean[..., 6:8])
        pm, pc = kf.predict(mean, self.cov)
        pm[..., 2:4] = pm[..., 2:4].clamp(min=1.0)
        self.mean = torch.where(live_t[..., None], pm, self.mean)
        self.cov = torch.where(live_t[..., None, None], pc, self.cov)
        self.tsu[live] += 1
        self.tsg[live] += 1

        # ---- 2. candidate compaction + cost terms -------------------------------------------
        cand = [np.nonzero(live[s])[0] for s in range(S)]
        Mc = max(1, max(len(c) for c in cand))
        cidx = np.zeros((S, Mc), np.int64)
        cmask = np.zeros((S, Mc), bool)
        for s in range(S):
            cidx[s, :len(cand[s])] = cand[s]
            cmask[s, :len(cand[s])] = True
        cidx_t = torch.from_numpy(cidx).to(dev, non_blocking=True)
        g = lambda x: x.gather(1, cidx_t.view(S, Mc, *([1] * (x.dim() - 2))).expand(S, Mc, *x.shape[2:]))
        tm, tc = g(self.mean), g(self.cov)
        tbox = kf.xywh_to_xyxy(tm[..., :4])
        dz = kf.xyxy_to_xywh(boxes)
        iou = kf.pairwise_iou(tbox, boxes)                               # [S,Mc,N]
        d2 = kf.mahalanobis_sq(tm, tc, dz)                               # [S,Mc,N]
        lobs = kf.xywh_to_xyxy(g(self.last_obs))
        iou_last = kf.pairwise_iou(lobs, boxes)
        # OCM: direction (prev obs -> last obs) vs (prev obs -> det)
        hist = g(self.hist)                                              # [S,Mc,K,4]
        k_prev = torch.from_numpy(np.minimum(np.take_along_axis(self.nobs, cidx, 1) - 1,
                                             cfg.ocm_delta).clip(0)).to(dev, non_blocking=True)
        prev = hist.gather(2, k_prev[..., None, None].expand(S, Mc, 1, 4))[:, :, 0, :2]
        v_trk = F.normalize(hist[:, :, 0, :2] - prev, dim=-1)
        v_det = F.normalize(dz[:, None, :, :2] - prev[:, :, None, :], dim=-1)
        cosang = (v_trk[:, :, None, :] * v_det).sum(-1).clamp(-1, 1)
        angle_sim = (np.pi / 2 - torch.arccos(cosang).abs()) / np.pi   # [-0.5, 0.5]
        if emb is not None and cfg.use_appearance:
            edist = 1.0 - torch.einsum("smd,snd->smn", g(self.emb), emb)
        else:
            edist = torch.ones_like(iou)
        # det-det overlap (for appearance update gating)
        dd = kf.pairwise_iou(boxes, boxes)
        dd = dd - torch.eye(N, device=dev) * 2
        dd = dd.masked_fill(~valid[:, None, :], 0).amax(-1)              # [S,N]
        packed = torch.stack([iou, d2.clamp(max=BIG), iou_last, angle_sim, edist], 1)
        host = packed.to("cpu", non_blocking=True)
        sv = torch.cat([torch.stack([scores, cls.float(), valid.float(), dd], -1), dz[..., 2:4]], -1)
        sv = sv.to("cpu", non_blocking=True)                              # [S,N,6]
        twh = tm[..., 2:4].to("cpu", non_blocking=True)                   # [S,Mc,2]
        torch.cuda.current_stream(dev).synchronize()
        host, sv, twh = host.numpy(), sv.numpy(), twh.numpy()
        if scores_np is None:
            scores_np, cls_np, valid_np = sv[..., 0], sv[..., 1].astype(np.int64), sv[..., 2] > 0.5
        dd_np, dwh = sv[..., 3], sv[..., 4:6]
        t1 = _now()

        # ---- 3. CPU matching cascade ---------------------------------------------------------
        m_s, m_slot, m_det, m_gap, m_good = [], [], [], [], []
        new_s, new_slot, new_det = [], [], []
        upd_emb_s, upd_emb_slot, upd_emb_det, upd_emb_a = [], [], [], []
        for s in range(S):
            if not active[s]:
                continue
            sl = cand[s]
            n = len(sl)
            sc, vd = scores_np[s], valid_np[s]
            dets_high = np.nonzero(vd & (sc >= cfg.high_thresh))[0]
            dets_low = np.nonzero(vd & (sc >= cfg.low_thresh) & (sc < cfg.high_thresh))[0]
            IOU, D2, IOUL, ANG, ED = (host[s, k, :n] for k in range(5))
            st, tsu = self.state[s, sl], self.tsu[s, sl]
            hasemb = self.has_emb[s, sl]
            if cfg.class_aware:
                tg = self._group(self.cls[s, sl])
                dg = self._group(cls_np[s])
                same = tg[:, None] == dg[None, :]
            else:
                same = np.ones((n, N), bool)
            matched_t = np.zeros(n, bool)
            used_d = np.zeros(N, bool)
            pairs = []  # (track_row, det, stage)

            def run(rows, cols, cost, limit, stage):
                rows = rows[~matched_t[rows]]
                cols = cols[~used_d[cols]]
                if len(rows) == 0 or len(cols) == 0:
                    return
                c = cost[np.ix_(rows, cols)]
                r, k = _lap(c, limit)
                for a, b in zip(rows[r], cols[k]):
                    matched_t[a] = True
                    used_d[b] = True
                    pairs.append((a, b, stage))

            conf_rows = np.nonzero((st == TRACKED) | (st == LOST))[0]
            # S1: fused IoU / appearance + OCM, gated
            iou_c = 1.0 - IOU
            emb_c = np.where(hasemb[:, None], ED, 1.0)
            emb_c = np.where((emb_c > cfg.emb_thresh) | (iou_c > cfg.proximity_thresh), 1.0, emb_c)
            c1 = np.minimum(iou_c, emb_c)
            has_vel = self.nobs[s, sl] >= 2
            c1 = c1 - cfg.ocm_weight * ANG * sc[None, :] * has_vel[:, None]
            c1 = np.where((D2 > cfg.gate_chi2) | ~same | (IOU <= 0) & (emb_c >= 1.0), BIG, c1)
            run(conf_rows, dets_high, c1, cfg.match_thresh, 1)
            # S2: BYTE low-score association, tracks seen last frame only
            c2 = np.where((D2 > cfg.gate_chi2) | ~same, BIG, iou_c)
            run(conf_rows[tsu[conf_rows] == 1], dets_low, c2, cfg.low_match_thresh, 2)
            # S3: OC-SORT OCR -- last observation vs remaining high dets
            c3 = np.where(~same, BIG, 1.0 - IOUL)
            run(conf_rows[tsu[conf_rows] <= cfg.ocr_max_gap], dets_high, c3, cfg.ocr_thresh, 3)
            # S4: long-term re-ID (appearance) for lost tracks
            if cfg.use_appearance and emb is not None:
                c4 = np.where((D2 > cfg.reid_gate_chi2) | ~same | ~hasemb[:, None], BIG, ED)
                run(conf_rows[tsu[conf_rows] > 1], dets_high, c4, cfg.reid_thresh, 4)
            # S5: tentative tracks
            tent_rows = np.nonzero(st == TENTATIVE)[0]
            c5 = np.where(~same, BIG, iou_c)
            run(tent_rows, dets_high, c5, cfg.unconfirmed_thresh, 5)

            for a, b, stage in pairs:
                slot = sl[a]
                ratio = (dwh[s, b] / np.maximum(twh[s, a], 1e-3)).min()
                good = bool(sc[b] >= cfg.high_thresh and ratio >= cfg.shape_change_ratio)
                gap = int(self.tsg[s, slot]) if good else 0
                if stage == 3:
                    self.stats["ocr"] += 1
                if stage == 4:
                    self.stats["reid"] += 1
                m_s.append(s); m_slot.append(slot); m_det.append(b); m_gap.append(gap); m_good.append(good)
                self.tsu[s, slot] = 0
                self.hits[s, slot] += 1
                if good:
                    self.tsg[s, slot] = 0
                    self.nobs[s, slot] += 1
                self.score[s, slot] = sc[b]
                if sc[b] >= cfg.high_thresh:
                    self.cls[s, slot] = cls_np[s, b]
                if self.state[s, slot] == TENTATIVE and self.hits[s, slot] >= cfg.n_init:
                    self.state[s, slot] = TRACKED
                    self.ids[s, slot] = self.next_id[s]
                    self.next_id[s] += 1
                elif self.state[s, slot] == LOST:
                    self.state[s, slot] = TRACKED
                if emb is not None and good and dd_np[s, b] < cfg.emb_update_max_overlap:
                    # dynamic EMA: trust confident detections more
                    trust = (sc[b] - cfg.high_thresh) / (1 - cfg.high_thresh)
                    a_ = cfg.emb_alpha + (1 - cfg.emb_alpha) * (1 - trust)
                    upd_emb_s.append(s); upd_emb_slot.append(slot); upd_emb_det.append(b)
                    upd_emb_a.append(a_ if self.has_emb[s, slot] else 0.0)
                    self.has_emb[s, slot] = True
            # unmatched tracks
            for a in np.nonzero(~matched_t)[0]:
                slot = sl[a]
                if self.state[s, slot] == TENTATIVE:
                    self.state[s, slot] = FREE
                elif self.state[s, slot] == TRACKED:
                    self.state[s, slot] = LOST
                if self.state[s, slot] == LOST and self.tsu[s, slot] > self.buffer[s]:
                    self.state[s, slot] = FREE
            # new tracks
            rem = [d for d in np.nonzero(vd & (sc >= cfg.new_track_thresh) & ~used_d)[0]]
            free = np.nonzero(self.state[s] == FREE)[0]
            for d, slot in zip(rem, free):
                new_s.append(s); new_slot.append(slot); new_det.append(d)
                first = self.frame[s] == 1
                self.state[s, slot] = TRACKED if first else TENTATIVE
                self.ids[s, slot] = -1
                if first:
                    self.ids[s, slot] = self.next_id[s]
                    self.next_id[s] += 1
                self.tsu[s, slot] = 0
                self.tsg[s, slot] = 0
                self.hits[s, slot] = 1
                self.nobs[s, slot] = 1
                self.cls[s, slot] = cls_np[s, d]
                self.score[s, slot] = sc[d]
                self.has_emb[s, slot] = emb is not None and dd_np[s, d] < cfg.emb_update_max_overlap
                if self.has_emb[s, slot]:
                    upd_emb_s.append(s); upd_emb_slot.append(slot); upd_emb_det.append(d); upd_emb_a.append(0.0)
        t2 = _now()

        # ---- 4. GPU updates -----------------------------------------------------------------
        T = lambda x: torch.as_tensor(np.asarray(x, np.int64)).to(dev, non_blocking=True)
        if m_s:
            ms, mslot, mdet = T(m_s), T(m_slot), T(m_det)
            gaps = np.asarray(m_gap)
            good = np.asarray(m_good)
            z = dz[ms, mdet]
            zs = scores[ms, mdet]
            rsc = torch.as_tensor(np.where(good, 1.0, cfg.low_quality_r_scale).astype(np.float32)).to(dev)
            mean, cov = self.mean[ms, mslot], self.cov[ms, mslot]
            um, uc = kf.update(mean, cov, z, zs, rsc)
            if cfg.use_oru and (gaps > 1).any():
                ri = np.nonzero(gaps > 1)[0]
                self.stats["oru"] += len(ri)
                ri_t = T(ri)
                rm, rc = self._oru(self.snap_mean[ms[ri_t], mslot[ri_t]], self.snap_cov[ms[ri_t], mslot[ri_t]],
                                   self.last_obs[ms[ri_t], mslot[ri_t]], z[ri_t], zs[ri_t], gaps[ri])
                um[ri_t], uc[ri_t] = rm, rc
            um[..., 2:4] = um[..., 2:4].clamp(min=1.0)
            self.mean[ms, mslot] = um
            self.cov[ms, mslot] = uc
            if good.any():
                gi = T(np.nonzero(good)[0])
                gs, gslot, zg = ms[gi], mslot[gi], z[gi]
                self.snap_mean[gs, gslot] = um[gi]
                self.snap_cov[gs, gslot] = uc[gi]
                self.last_obs[gs, gslot] = zg
                self.hist[gs, gslot] = torch.cat([zg[:, None], self.hist[gs, gslot, :-1]], 1)
        if new_s:
            ns, nslot, ndet = T(new_s), T(new_slot), T(new_det)
            z = dz[ns, ndet]
            nm, nc = kf.initiate(z)
            self.mean[ns, nslot] = nm
            self.cov[ns, nslot] = nc
            self.snap_mean[ns, nslot] = nm
            self.snap_cov[ns, nslot] = nc
            self.last_obs[ns, nslot] = z
            self.hist[ns, nslot] = z[:, None].expand(-1, self.hist.shape[2], -1)
        if upd_emb_s and emb is not None:
            es, eslot, edet = T(upd_emb_s), T(upd_emb_slot), T(upd_emb_det)
            a = torch.as_tensor(np.asarray(upd_emb_a, np.float32)).to(dev, non_blocking=True)[:, None]
            e = a * self.emb[es, eslot] + (1 - a) * emb[es, edet]
            self.emb[es, eslot] = F.normalize(e, dim=-1)

        # ---- outputs: confirmed tracks updated this frame -------------------------------------
        outs: list[FrameOutput | None] = []
        out_mask = (self.state == TRACKED) & (self.tsu == 0)
        for s in range(S):
            if not active[s]:
                outs.append(None)
                continue
            slots = np.nonzero(out_mask[s])[0]
            ob = kf.xywh_to_xyxy(self.mean[s, T(slots), :4]) if len(slots) else torch.zeros(0, 4, device=dev)
            outs.append(FrameOutput(self.ids[s, slots].copy(), ob, self.score[s, slots].copy(),
                                    self.cls[s, slots].copy()))
        t3 = _now()
        self.timing["gpu_cost"] += t1 - t0
        self.timing["cpu_match"] += t2 - t1
        self.timing["gpu_update"] += t3 - t2
        return outs

    # ------------------------------------------------------------------------------------------
    def _oru(self, m0, c0, z0, z1, score, gaps):
        """OC-SORT observation-centric re-update along a virtual linear trajectory.

        m0/c0: posterior at the last real observation z0 (xywh); z1: new observation after
        ``gaps`` frames.  Re-runs the filter from the last observation with virtual observations
        linearly interpolated between z0 and z1.  For gaps longer than ``oru_max_steps`` the
        virtual observations are placed at ``oru_max_steps`` evenly spaced times and the filter
        jumps between them with exact multi-step predictions (bounded cost per re-acquisition).
        """
        K = self.cfg.oru_max_steps
        dev = m0.device
        g = torch.as_tensor(gaps, device=dev, dtype=torch.float32)
        n_upd = torch.clamp(g, max=K)                 # number of virtual/real updates per track
        m, c = m0.clone(), c0.clone()
        t_prev = torch.zeros_like(g)
        for k in range(1, int(n_upd.max().item()) + 1):
            act = k <= n_upd
            t_k = torch.round(k * g / n_upd)
            dt = torch.where(act, t_k - t_prev, torch.ones_like(g))
            pm, pc = kf.predict_dt(m, c, dt)
            zi = z0 + (z1 - z0) * (t_k / g)[:, None]
            um, uc = kf.update(pm, pc, zi, score)
            m = torch.where(act[:, None], um, m)
            c = torch.where(act[:, None, None], uc, c)
            t_prev = torch.where(act, t_k, t_prev)
        return m, c


def _now():
    import time
    return time.perf_counter()
