"""Batched multi-stream tracker: Kalman (GPU) + BYTE + OC-SORT (ORU/OCM/OCR) + appearance re-ID.

Data layout
-----------
Numeric track state lives on the GPU in fixed-capacity slot tensors ``[S, M, ...]`` (S streams,
M slots per stream); detections are padded to ``[S, N]``.  Because every shape is static, the GPU
part of a step is two CUDA-graph replays (no per-kernel Python launch overhead):

  graph A (pre-match):  KF predict of every live slot, then *all* cost terms of the cascade for
                        every slot x detection pair: IoU(pred, det), Mahalanobis d^2, IoU(last
                        observation, det), OCM direction term, appearance cosine distance; plus
                        det-det overlap.  Copied into pinned host memory inside the graph.
  CPU                :  matching cascade per stream with ``lap.lapjv`` on the live rows only
                        (small integer bookkeeping -- ids, states, ages -- lives on the CPU):
       S1  confirmed (tracked + lost) x high dets   fused IoU/appearance + OCM, Mahalanobis gate
       S2  tracked-last-frame x low dets             IoU (BYTE second association)
       S3  remaining x remaining high dets           IoU(last observation, det) (OC-SORT OCR)
       S4  lost x remaining high dets                appearance only, loose Mahalanobis gate (re-ID)
       S5  tentative x remaining high dets           IoU
       new tentative tracks from leftover high dets
  graph B (post-match): dense masked KF update of matched slots (NSA noise, inflated R for
                        low-quality boxes), OC-SORT observation-centric re-update (ORU) for slots
                        re-matched after a gap, track initiation, appearance EMA, output boxes.

``cuda_graph=False`` runs the identical code eagerly (debugging).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import lap
import numpy as np
import torch
import torch.nn.functional as F

from . import kalman as kf

FREE, TENTATIVE, TRACKED, LOST = 0, 1, 2, 3
BIG = 6e4  # fits in fp16

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
    oru_max_steps: int = 8          # virtual observations per re-update (longer gaps are subsampled)
    freeze_size_when_lost: bool = True
    # low-quality observations (score < high_thresh, or box abruptly shrinks = partial occlusion):
    # inflate R, and do not use them as OC-SORT "observations" (snapshot / last_obs / OCM history)
    low_quality_r_scale: float = 100.0
    shape_change_ratio: float = 0.75
    class_aware: bool = True
    class_groups: dict = field(default_factory=lambda: dict(DEFAULT_CLASS_GROUPS))
    max_tracks: int = 160           # slots per stream (M)
    max_dets: int = 160             # padded detections per frame (N)
    min_box_area: float = 10.0
    cuda_graph: bool = True
    compile: bool = True            # torch.compile (inductor) the dense GPU parts before graph capture


@dataclass
class FrameOutput:
    ids: np.ndarray       # [n] int64
    boxes: np.ndarray     # [n,4] xyxy (KF posterior)
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
        c = self.cfg
        self.S, self.M, self.N, self.D = num_streams, c.max_tracks, c.max_dets, emb_dim
        self.dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        fps = fps if isinstance(fps, (list, tuple)) else [fps] * num_streams
        self.buffer = np.array([max(1, int(round(f * c.lost_buffer_s))) for f in fps])
        S, M, N, D, dev = self.S, self.M, self.N, self.D, self.dev
        cuda = self.dev.type == "cuda"
        pin = dict(pin_memory=cuda)
        # ---- GPU numeric state (static addresses: updated in place) ----
        self.mean = torch.zeros(S, M, 8, device=dev)
        self.cov = torch.zeros(S, M, 8, 8, device=dev)
        self.snap_mean = torch.zeros(S, M, 8, device=dev)    # posterior at last observation (ORU)
        self.snap_cov = torch.zeros(S, M, 8, 8, device=dev)
        self.last_obs = torch.zeros(S, M, 4, device=dev)      # xywh
        self.hist = torch.zeros(S, M, c.ocm_delta + 1, 4, device=dev)  # xywh, newest first
        self.emb = torch.zeros(S, M, D, device=dev)
        # ---- static inputs ----
        self.in_boxes = torch.zeros(S, N, 4, device=dev)
        self.in_scores = torch.zeros(S, N, device=dev)
        self.in_valid = torch.zeros(S, N, dtype=torch.bool, device=dev)
        self.in_emb = torch.zeros(S, N, D, device=dev)
        self.dz = torch.zeros(S, N, 4, device=dev)
        self.pre_dev = torch.zeros(3, S, M, device=dev)       # live, lost, k_prev
        self.post_dev = torch.zeros(8, S, M, device=dev)      # see _post
        self.pre_host = torch.zeros(3, S, M, **pin)
        self.post_host = torch.zeros(8, S, M, **pin)
        # ---- static outputs (pinned host) ----
        self.cost_host = torch.zeros(S, 5, M, N, dtype=torch.float16, **pin)
        self.det_host = torch.zeros(S, N, 3, **pin)           # det-det max overlap, det w, det h
        self.twh_host = torch.zeros(S, M, 2, **pin)
        self.out_host = torch.zeros(S, M, 4, **pin)
        self._reset_gpu_state()
        # ---- CPU bookkeeping ----
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
        lut = np.arange(1000, 1000 + 4096, dtype=np.int64)
        for k, v in c.class_groups.items():
            lut[k] = v
        self._group_lut = lut
        self.timing = {"gpu_pre": 0.0, "cpu_match": 0.0, "gpu_post": 0.0}
        self.stats = {"oru": 0, "reid": 0, "ocr": 0, "dropped_dets": 0, "slots_full": 0}
        self.graphs = {}
        if c.compile and cuda:
            kf.UNROLLED_4X4 = True  # element-wise 4x4 Cholesky: fused by inductor
            self._pre_core = torch.compile(self._pre_core, dynamic=False)
            self._post_core = torch.compile(self._post_core, dynamic=False)
            self._post_core_oru = torch.compile(self._post_core_oru, dynamic=False)
        if c.cuda_graph and cuda:
            self._capture()

    # ------------------------------------------------------------------------------------------
    def _reset_gpu_state(self):
        self.mean.zero_()
        self.mean[..., 2:4] = 1.0
        self.cov.copy_(torch.eye(8, device=self.dev).expand_as(self.cov))
        self.snap_mean.copy_(self.mean)
        self.snap_cov.copy_(self.cov)
        self.last_obs.zero_()
        self.last_obs[..., 2:4] = 1.0
        self.hist.zero_()
        self.emb.zero_()

    def _capture(self):
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(2):  # warm-up (cuBLAS / cuSOLVER handles, allocator)
                self._pre()
                self._post(True)
                self._post(False)
        torch.cuda.current_stream(self.dev).wait_stream(s)
        for name, fn in (("pre", self._pre), ("post", lambda: self._post(False)),
                         ("post_oru", lambda: self._post(True))):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                fn()
            self.graphs[name] = g
        torch.cuda.synchronize(self.dev)
        self._reset_gpu_state()

    def _run(self, name, fn):
        if name in self.graphs:
            self.graphs[name].replay()
        else:
            fn()

    # ------------------------------------------------------------------------------------------
    def _pre(self):
        """graph A: predict + cost terms (dense over all slots), then pinned D2H copies."""
        packed, det, twh = self._pre_core()
        self.cost_host.copy_(packed, non_blocking=True)
        self.det_host.copy_(det, non_blocking=True)
        self.twh_host.copy_(twh, non_blocking=True)

    def _pre_core(self):
        c = self.cfg
        live = self.pre_dev[0] > 0.5
        lost = self.pre_dev[1] > 0.5
        kprev = self.pre_dev[2].long()
        mean = self.mean
        if c.freeze_size_when_lost:
            mean = mean.clone()
            mean[..., 6:8] = torch.where(lost[..., None], 0.0, mean[..., 6:8])
        pm, pc = kf.predict(mean, self.cov)
        pm[..., 2:4] = pm[..., 2:4].clamp(min=1.0)
        self.mean.copy_(torch.where(live[..., None], pm, self.mean))
        self.cov.copy_(torch.where(live[..., None, None], pc, self.cov))
        boxes = self.in_boxes
        dz = kf.xyxy_to_xywh(boxes)
        self.dz.copy_(dz)
        tbox = kf.xywh_to_xyxy(self.mean[..., :4])
        iou = kf.pairwise_iou(tbox, boxes)                               # [S,M,N]
        d2 = kf.mahalanobis_sq(self.mean, self.cov, dz)
        iou_last = kf.pairwise_iou(kf.xywh_to_xyxy(self.last_obs), boxes)
        # OCM: direction (prev obs -> last obs) vs (prev obs -> det)
        S, M = kprev.shape
        prev = self.hist.gather(2, kprev[..., None, None].expand(S, M, 1, 4))[:, :, 0, :2]
        v_trk = F.normalize(self.hist[:, :, 0, :2] - prev, dim=-1)
        v_det = F.normalize(dz[:, None, :, :2] - prev[:, :, None, :], dim=-1)
        cosang = (v_trk[:, :, None, :] * v_det).sum(-1).clamp(-1, 1)
        angle_sim = (np.pi / 2 - torch.arccos(cosang).abs()) / np.pi    # [-0.5, 0.5]
        edist = 1.0 - torch.einsum("smd,snd->smn", self.emb, self.in_emb)
        packed = torch.stack([iou, d2, iou_last, angle_sim, edist], 1)
        packed = torch.nan_to_num(packed, nan=BIG, posinf=BIG, neginf=-BIG).clamp(-BIG, BIG)
        dd = kf.pairwise_iou(boxes, boxes)
        eye = torch.eye(boxes.shape[1], device=boxes.device, dtype=torch.bool)
        dd = dd.masked_fill(eye | ~self.in_valid[:, None, :], 0).amax(-1)
        return packed.half(), torch.cat([dd[..., None], dz[..., 2:4]], -1), self.mean[..., 2:4].clone()

    def _post(self, with_oru: bool):
        """graph B: dense masked update / ORU / initiation / appearance EMA, then output D2H."""
        out = (self._post_core_oru if with_oru else self._post_core)()
        self.out_host.copy_(out, non_blocking=True)

    def _post_core_oru(self):
        return self._post_impl(True)

    def _post_core(self):
        return self._post_impl(False)

    def _post_impl(self, with_oru: bool):
        c = self.cfg
        p = self.post_dev
        matched, mdet, good, gap = p[0] > 0.5, p[1].long(), p[2] > 0.5, p[3]
        new, ndet, emb_upd, emb_a = p[4] > 0.5, p[5].long(), p[6] > 0.5, p[7]
        S, M = mdet.shape
        g4 = lambda x, i: x.gather(1, i[..., None].expand(S, M, x.shape[-1]))
        z = g4(self.dz, mdet)
        zs = self.in_scores.gather(1, mdet)
        rsc = torch.where(good, 1.0, c.low_quality_r_scale)
        um, uc = kf.update(self.mean, self.cov, z, zs, rsc)
        if with_oru:
            rm, rc = self._oru_dense(z, zs, gap)
            oru = gap > 1.5
            um = torch.where(oru[..., None], rm, um)
            uc = torch.where(oru[..., None, None], rc, uc)
        um[..., 2:4] = um[..., 2:4].clamp(min=1.0)
        nz = g4(self.dz, ndet)
        nm, nc = kf.initiate(nz)
        mean = torch.where(matched[..., None], um, self.mean)
        cov = torch.where(matched[..., None, None], uc, self.cov)
        mean = torch.where(new[..., None], nm, mean)
        cov = torch.where(new[..., None, None], nc, cov)
        self.mean.copy_(mean)
        self.cov.copy_(cov)
        obs = (matched & good) | new
        zz = torch.where(new[..., None], nz, z)
        self.snap_mean.copy_(torch.where(obs[..., None], mean, self.snap_mean))
        self.snap_cov.copy_(torch.where(obs[..., None, None], cov, self.snap_cov))
        self.last_obs.copy_(torch.where(obs[..., None], zz, self.last_obs))
        shifted = torch.cat([zz[:, :, None], self.hist[:, :, :-1]], 2)
        filled = zz[:, :, None].expand_as(self.hist)
        newhist = torch.where(new[..., None, None], filled, shifted)
        self.hist.copy_(torch.where(obs[..., None, None], newhist, self.hist))
        edet = torch.where(new, ndet, mdet)
        e = emb_a[..., None] * self.emb + (1 - emb_a[..., None]) * g4(self.in_emb, edet)
        self.emb.copy_(torch.where(emb_upd[..., None], F.normalize(e, dim=-1), self.emb))
        return kf.xywh_to_xyxy(self.mean[..., :4])

    def _oru_dense(self, z1, score, gap):
        """OC-SORT observation-centric re-update along a virtual linear trajectory (dense, masked).

        Restart from the posterior at the last trusted observation z0 and re-run the filter with
        virtual observations interpolated linearly between z0 and the new detection z1.  Gaps
        longer than ``oru_max_steps`` get that many evenly spaced virtual observations with exact
        multi-step predictions in between, so the cost per step is fixed (graph-capturable).
        """
        K = self.cfg.oru_max_steps
        g = gap.clamp(min=1.0)
        n_upd = g.clamp(max=K)
        z0 = self.last_obs
        m, cv = self.snap_mean, self.snap_cov
        t_prev = torch.zeros_like(g)
        for k in range(1, K + 1):
            act = k <= n_upd
            t_k = torch.round(k * g / n_upd)
            dt = torch.where(act, t_k - t_prev, torch.ones_like(g))
            pm, pc = kf.predict_dt(m, cv, dt)
            zi = z0 + (z1 - z0) * (t_k / g)[..., None]
            um, uc = kf.update(pm, pc, zi, score)
            m = torch.where(act[..., None], um, m)
            cv = torch.where(act[..., None, None], uc, cv)
            t_prev = torch.where(act, t_k, t_prev)
        return m, cv

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

    def _load_inputs(self, boxes, scores, valid, emb):
        n = min(boxes.shape[1], self.N)
        if boxes.shape[1] > self.N:
            self.stats["dropped_dets"] += int(valid[:, self.N:].sum())
        self.in_boxes[:, :n].copy_(boxes[:, :n])
        self.in_scores[:, :n].copy_(scores[:, :n])
        self.in_valid[:, :n].copy_(valid[:, :n])
        if n < self.N:
            self.in_boxes[:, n:] = 0
            self.in_scores[:, n:] = 0
            self.in_valid[:, n:] = False
        if emb is not None:
            self.in_emb[:, :n].copy_(emb[:, :n])
            if n < self.N:
                self.in_emb[:, n:] = 0
        else:
            self.in_emb.zero_()
        return n

    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, boxes: torch.Tensor, scores: torch.Tensor, cls: torch.Tensor, valid: torch.Tensor,
             emb: torch.Tensor | None = None, active: np.ndarray | None = None,
             scores_np: np.ndarray | None = None, cls_np: np.ndarray | None = None,
             valid_np: np.ndarray | None = None) -> list[FrameOutput | None]:
        """One frame for every stream.

        boxes [S,n,4] xyxy (source px), scores [S,n], cls [S,n], valid [S,n] (GPU tensors; n is
        padded/truncated to ``max_dets``), emb [S,n,D] unit vectors or None.  ``active[s]=False``
        skips stream s (no frame).  ``*_np``: optional host copies of scores/cls/valid.
        """
        cfg, S = self.cfg, self.S
        use_emb = emb is not None and cfg.use_appearance
        active = np.ones(S, bool) if active is None else np.asarray(active, bool)
        t0 = time.perf_counter()
        n_in = self._load_inputs(boxes, scores, valid, emb if use_emb else None)
        if scores_np is None:
            scores_np = scores[:, :n_in].cpu().numpy()
            cls_np = cls[:, :n_in].cpu().numpy()
            valid_np = valid[:, :n_in].cpu().numpy()
        else:
            scores_np, cls_np, valid_np = scores_np[:, :n_in], cls_np[:, :n_in], valid_np[:, :n_in]
        live = (self.state != FREE) & active[:, None]
        self.frame[active] += 1
        ph = self.pre_host.numpy()
        ph[0] = live
        ph[1] = (self.state == LOST) & live
        ph[2] = np.minimum(self.nobs - 1, cfg.ocm_delta).clip(0)
        self.pre_dev.copy_(self.pre_host, non_blocking=True)
        self._run("pre", self._pre)
        torch.cuda.current_stream(self.dev).synchronize()
        self.tsu[live] += 1
        self.tsg[live] += 1
        host = self.cost_host.numpy()
        dh = self.det_host.numpy()
        dd_np, dwh = dh[..., 0], dh[..., 1:3]
        twh = self.twh_host.numpy()
        t1 = time.perf_counter()

        # ---- CPU matching cascade -------------------------------------------------------------
        post = self.post_host.numpy()
        post[:] = 0
        any_oru = False
        groups = self._group_lut[np.clip(cls_np, 0, 4095)] if cfg.class_aware else np.zeros_like(cls_np)
        for s in range(S):
            if not active[s]:
                continue
            sl = np.nonzero(live[s])[0]
            n = len(sl)
            sc, vd = scores_np[s], valid_np[s]
            Nn = len(sc)
            dets_high = np.nonzero(vd & (sc >= cfg.high_thresh))[0]
            dets_low = np.nonzero(vd & (sc >= cfg.low_thresh) & (sc < cfg.high_thresh))[0]
            pairs = []  # (track_row, det, stage)
            matched_t = np.zeros(n, bool)
            used_d = np.zeros(Nn, bool)
            if n:
                IOU, D2, IOUL, ANG, ED = host[s][:, sl, :Nn].astype(np.float32)
                st, tsu = self.state[s, sl], self.tsu[s, sl]
                hasemb = self.has_emb[s, sl] & use_emb
                tg = self._group_lut[self.cls[s, sl]] if cfg.class_aware else np.zeros(n, np.int64)
                same = tg[:, None] == groups[s][None, :]

                def run(rows, cols, cost, limit, stage):
                    rows = rows[~matched_t[rows]]
                    cols = cols[~used_d[cols]]
                    if len(rows) == 0 or len(cols) == 0:
                        return
                    r, k = _lap(cost[np.ix_(rows, cols)], limit)
                    for a, b in zip(rows[r], cols[k]):
                        matched_t[a] = True
                        used_d[b] = True
                        pairs.append((a, b, stage))

                conf_rows = np.nonzero((st == TRACKED) | (st == LOST))[0]
                # S1: fused IoU / appearance (BoT-SORT) + OCM, Mahalanobis-gated
                iou_c = 1.0 - IOU
                emb_c = np.where(hasemb[:, None], ED, 1.0)
                emb_c = np.where((emb_c > cfg.emb_thresh) | (iou_c > cfg.proximity_thresh), 1.0, emb_c)
                c1 = np.minimum(iou_c, emb_c)
                has_vel = self.nobs[s, sl] >= 2
                c1 = c1 - cfg.ocm_weight * ANG * sc[None, :] * has_vel[:, None]
                c1 = np.where((D2 > cfg.gate_chi2) | ~same | ((IOU <= 0) & (emb_c >= 1.0)), BIG, c1)
                run(conf_rows, dets_high, c1, cfg.match_thresh, 1)
                # S2: BYTE low-score association, tracks seen last frame only
                c2 = np.where((D2 > cfg.gate_chi2) | ~same, BIG, iou_c)
                run(conf_rows[tsu[conf_rows] == 1], dets_low, c2, cfg.low_match_thresh, 2)
                # S3: OC-SORT OCR -- last observation vs remaining high dets
                c3 = np.where(~same, BIG, 1.0 - IOUL)
                run(conf_rows[tsu[conf_rows] <= cfg.ocr_max_gap], dets_high, c3, cfg.ocr_thresh, 3)
                # S4: long-term re-ID (appearance) for lost tracks
                if use_emb:
                    c4 = np.where((D2 > cfg.reid_gate_chi2) | ~same | ~hasemb[:, None], BIG, ED)
                    run(conf_rows[tsu[conf_rows] > 1], dets_high, c4, cfg.reid_thresh, 4)
                # S5: tentative tracks
                c5 = np.where(~same, BIG, iou_c)
                run(np.nonzero(st == TENTATIVE)[0], dets_high, c5, cfg.unconfirmed_thresh, 5)

            for a, b, stage in pairs:
                slot = sl[a]
                ratio = (dwh[s, b] / np.maximum(twh[s, slot], 1e-3)).min()
                good = bool(sc[b] >= cfg.high_thresh and ratio >= cfg.shape_change_ratio)
                gap = int(self.tsg[s, slot]) if good else 0
                self.stats["ocr"] += stage == 3
                self.stats["reid"] += stage == 4
                post[0, s, slot] = 1
                post[1, s, slot] = b
                post[2, s, slot] = good
                if cfg.use_oru and gap > 1:
                    post[3, s, slot] = gap
                    any_oru = True
                    self.stats["oru"] += 1
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
                if use_emb and good and dd_np[s, b] < cfg.emb_update_max_overlap:
                    trust = (sc[b] - cfg.high_thresh) / (1 - cfg.high_thresh)   # dynamic EMA
                    post[6, s, slot] = 1
                    post[7, s, slot] = (cfg.emb_alpha + (1 - cfg.emb_alpha) * (1 - trust)) \
                        if self.has_emb[s, slot] else 0.0
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
            rem = np.nonzero(vd & (sc >= cfg.new_track_thresh) & ~used_d)[0]
            free = np.nonzero(self.state[s] == FREE)[0]
            if len(rem) > len(free):
                self.stats["slots_full"] += len(rem) - len(free)
            first = self.frame[s] == 1
            for d, slot in zip(rem, free):
                post[4, s, slot] = 1
                post[5, s, slot] = d
                self.state[s, slot] = TRACKED if first else TENTATIVE
                self.ids[s, slot] = -1
                if first:
                    self.ids[s, slot] = self.next_id[s]
                    self.next_id[s] += 1
                self.tsu[s, slot] = self.tsg[s, slot] = 0
                self.hits[s, slot] = self.nobs[s, slot] = 1
                self.cls[s, slot] = cls_np[s, d]
                self.score[s, slot] = sc[d]
                self.has_emb[s, slot] = use_emb and dd_np[s, d] < cfg.emb_update_max_overlap
                if self.has_emb[s, slot]:
                    post[6, s, slot] = 1
                    post[7, s, slot] = 0.0
        t2 = time.perf_counter()

        # ---- GPU post-match update ------------------------------------------------------------
        self.post_dev.copy_(self.post_host, non_blocking=True)
        if any_oru:
            self._run("post_oru", lambda: self._post(True))
        else:
            self._run("post", lambda: self._post(False))
        torch.cuda.current_stream(self.dev).synchronize()
        ob = self.out_host.numpy()
        outs: list[FrameOutput | None] = []
        out_mask = (self.state == TRACKED) & (self.tsu == 0)
        for s in range(S):
            if not active[s]:
                outs.append(None)
                continue
            slots = np.nonzero(out_mask[s])[0]
            outs.append(FrameOutput(self.ids[s, slots].copy(), ob[s, slots].copy(), self.score[s, slots].copy(),
                                    self.cls[s, slots].copy()))
        t3 = time.perf_counter()
        self.timing["gpu_pre"] += t1 - t0
        self.timing["cpu_match"] += t2 - t1
        self.timing["gpu_post"] += t3 - t2
        return outs
