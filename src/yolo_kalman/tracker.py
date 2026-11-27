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
from .assign import lap_batch

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
    lost_buffer_s: float = 8.0      # how long lost tracks coast before deletion (seconds)
    ocr_max_gap: int = 30           # S3 only for tracks lost <= this many frames
    n_init: int = 2                 # hits to confirm a tentative track
    # appearance
    use_appearance: bool = True
    emb_alpha: float = 0.9          # EMA factor (dynamic, Deep OC-SORT style)
    emb_thresh: float = 0.25        # S1: use appearance if cos-dist below this ...
    proximity_thresh: float = 0.5   # ... and 1-IoU below this (BoT-SORT fusion)
    reid_thresh: float = 0.25       # S4: max cos-dist for re-identification
    emb_update_max_overlap: float = 0.4  # skip EMA update when det overlaps another det (occluder)
    # id recovery: a *young* confirmed track (born while some track was lost) whose embedding
    # matches that lost track inherits the old id (the re-appearing object is often first seen
    # only partially, so the S4 re-ID match can fail during its first frames)
    id_recovery: bool = True
    young_min_frames: int = 3
    young_max_frames: int = 45
    # OC-SORT
    ocm_weight: float = 0.2
    ocm_delta: int = 3
    use_oru: bool = True
    oru_max_steps: int = 8          # virtual observations per re-update (longer gaps are subsampled)
    freeze_size_when_lost: bool = True
    # low-quality observations (score < high_thresh, or box abruptly shrinks = partial occlusion):
    # inflate R, and do not use them as OC-SORT "observations" (snapshot / last_obs / OCM history)
    low_quality_r_scale: float = 1e4
    shape_change_ratio: float = 0.9    # min det size / reference size (w and h) for a trusted obs
    size_ema: float = 0.9              # reference size = slow EMA over trusted observations (resists
                                       # gradual erosion of the box as an object slides behind an occluder)
    low_quality_size_r_scale: float = 1e4  # ... and do not let them change the box size
    class_aware: bool = True
    class_groups: dict = field(default_factory=lambda: dict(DEFAULT_CLASS_GROUPS))
    max_tracks: int = 160           # slots per stream (M)
    max_dets: int = 160             # padded detections per frame (N)
    min_box_area: float = 10.0
    cuda_graph: bool = True
    compile: bool = True            # torch.compile (inductor) the dense GPU parts before graph capture
    # assignment solver: "gpu_hungarian" = whole cascade on the GPU with the batched exact Hungarian
    # kernel (assign.lap_batch); "lapjv" = reference CPU cascade with lap.lapjv per stream and stage
    assign: str = "gpu_hungarian"
    max_pending: int = 64           # gpu_hungarian: max submitted-but-uncollected steps


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
        self._lq_scale = torch.tensor([c.low_quality_r_scale] * 2 + [c.low_quality_size_r_scale] * 2, device=dev)
        self.pre_dev = torch.zeros(3, S, M, device=dev)       # live, lost, k_prev
        self.post_dev = torch.zeros(8, S, M, device=dev)      # see _post
        self.pre_host = torch.zeros(3, S, M, **pin)
        self.post_host = torch.zeros(8, S, M, **pin)
        # ---- static outputs (pinned host) ----
        self.cost_host = torch.zeros(S, 5, M, N, dtype=torch.float16, **pin)
        self.det_host = torch.zeros(S, N, 3, **pin)           # det-det max overlap, det w, det h
        self.twh_host = torch.zeros(S, M, 4, **pin)
        self.tt_host = torch.zeros(S, 2, M, M, dtype=torch.float16, **pin)  # track-track emb dist, d2            # predicted w,h ; last trusted obs w,h
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
        self.ref_wh = np.ones((S, M, 2), np.float32)  # slow EMA of trusted box size (w, h)
        self.birth = np.zeros((S, M), np.int64)       # frame the slot's track was born
        self.emb_count = np.zeros((S, M), np.int32)   # appearance updates so far
        self.cls = np.zeros((S, M), np.int64)
        self.score = np.zeros((S, M), np.float32)
        self.next_id = np.ones(S, np.int64)
        self.frame = np.zeros(S, np.int64)
        lut = np.arange(1000, 1000 + 4096, dtype=np.int64)
        for k, v in c.class_groups.items():
            lut[k] = v
        self._group_lut = lut
        self.timing = {"gpu_pre": 0.0, "match": 0.0, "gpu_post": 0.0}
        self.stats = {"oru": 0, "reid": 0, "ocr": 0, "id_recovered": 0, "dropped_dets": 0, "slots_full": 0}
        if c.assign not in ("gpu_hungarian", "lapjv"):
            raise ValueError(f"assign must be 'gpu_hungarian' or 'lapjv', got {c.assign!r}")
        self.gpu_assign = c.assign == "gpu_hungarian" and cuda  # lap_batch is CUDA-only: CPU uses lapjv
        if self.gpu_assign:
            self._init_gpu_cascade(fps)
        self.graphs = {}
        if c.compile and cuda:
            kf.UNROLLED_4X4 = True  # element-wise 4x4 Cholesky: fused by inductor
            if self.gpu_assign:
                self._gpre_core = torch.compile(self._gpre_core, dynamic=False)
                self._gmatch_core = torch.compile(self._gmatch_core, dynamic=False)
                self._gpost_core = torch.compile(self._gpost_core, dynamic=False)
            else:
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
        if self.gpu_assign:
            fns = (("gpre", self._gpre), ("gmatch", self._gmatch), ("gpost", self._gpost))
        else:
            fns = (("pre", self._pre), ("post", lambda: self._post(False)), ("post_oru", lambda: self._post(True)))
        with torch.cuda.stream(s):
            for _ in range(2):  # warm-up (cuBLAS / cuSOLVER handles, allocator, CUDA extension build)
                for _, fn in fns:
                    fn()
        torch.cuda.current_stream(self.dev).wait_stream(s)
        for name, fn in fns:
            g = torch.cuda.CUDAGraph()
            # thread_local: NVDEC decoder threads keep issuing CUDA calls while we capture
            with torch.cuda.graph(g, stream=s, capture_error_mode="thread_local"):
                fn()
            self.graphs[name] = g
        torch.cuda.synchronize(self.dev)
        self._reset_gpu_state()
        if self.gpu_assign:
            self._reset_gpu_book()

    def _run(self, name, fn):
        if name in self.graphs:
            self.graphs[name].replay()
        else:
            fn()

    # ------------------------------------------------------------------------------------------
    def _pre(self):
        """graph A: predict + cost terms (dense over all slots), then pinned D2H copies."""
        packed, det, twh, tt = self._pre_core()
        self.cost_host.copy_(packed, non_blocking=True)
        self.tt_host.copy_(tt, non_blocking=True)
        self.det_host.copy_(det, non_blocking=True)
        self.twh_host.copy_(twh, non_blocking=True)

    def _pre_core(self):
        return self._pre_terms(self.pre_dev[0] > 0.5, self.pre_dev[1] > 0.5, self.pre_dev[2].long())

    def _pre_terms(self, live, lost, kprev):
        c = self.cfg
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
        # fp16 tensor-core matmul (fp32 accumulate); the result is stored as fp16 anyway
        edist = 1.0 - torch.einsum("smd,snd->smn", self.emb.half(), self.in_emb.half()).float()
        packed = torch.stack([iou, d2, iou_last, angle_sim, edist], 1)
        packed = torch.nan_to_num(packed, nan=BIG, posinf=BIG, neginf=-BIG).clamp(-BIG, BIG)
        dd = kf.pairwise_iou(boxes, boxes)
        eye = torch.eye(boxes.shape[1], device=boxes.device, dtype=torch.bool)
        dd = dd.masked_fill(eye | ~self.in_valid[:, None, :], 0).amax(-1)
        # track-track terms for id recovery: rows = (lost) track distribution, cols = (young) track box
        tt_e = 1.0 - torch.einsum("smd,snd->smn", self.emb.half(), self.emb.half()).float()
        tt_d2 = kf.mahalanobis_sq(self.mean, self.cov, self.mean[..., :4])
        tt = torch.nan_to_num(torch.stack([tt_e, tt_d2], 1), nan=BIG, posinf=BIG, neginf=-BIG).clamp(-BIG, BIG)
        return (packed.half(), torch.cat([dd[..., None], dz[..., 2:4]], -1),
                torch.cat([self.mean[..., 2:4], self.last_obs[..., 2:4]], -1), tt.half())

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
        lq = self._lq_scale
        rsc = torch.where(good[..., None], torch.ones_like(lq), lq)       # per-dimension R scale
        um, uc = kf.update(self.mean, self.cov, z, zs, rsc)
        if with_oru:
            rm, rc = self._oru_dense(z, zs, gap)
            oru = gap > 1.5
            um = torch.where(oru[..., None], rm, um)
            uc = torch.where(oru[..., None, None], rc, uc)
        um[..., 2:4] = um[..., 2:4].clamp(min=1.0)
        # low-quality (partially occluded) match: coast -- no size velocity either
        um[..., 6:8] = torch.where(good[..., None], um[..., 6:8], 0.0)
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

    # ==========================================================================================
    # assign="gpu_hungarian": the whole matching cascade + bookkeeping on the GPU
    # ==========================================================================================
    def _init_gpu_cascade(self, fps):
        S, M, N, dev = self.S, self.M, self.N, self.dev
        c = self.cfg
        cuda = dev.type == "cuda"
        i32 = dict(dtype=torch.int32, device=dev)
        i64 = dict(dtype=torch.int64, device=dev)
        self.g_state = torch.zeros(S, M, **i32)
        self.g_ids = torch.full((S, M), -1, **i64)
        self.g_tsu = torch.zeros(S, M, **i32)
        self.g_tsg = torch.zeros(S, M, **i32)
        self.g_hits = torch.zeros(S, M, **i32)
        self.g_nobs = torch.zeros(S, M, **i32)
        self.g_has_emb = torch.zeros(S, M, dtype=torch.bool, device=dev)
        self.g_ref_wh = torch.ones(S, M, 2, device=dev)
        self.g_birth = torch.zeros(S, M, **i64)
        self.g_emb_count = torch.zeros(S, M, **i32)
        self.g_cls = torch.zeros(S, M, **i64)
        self.g_score = torch.zeros(S, M, device=dev)
        self.g_next_id = torch.ones(S, **i64)
        self.g_frame = torch.zeros(S, **i64)
        self.g_buffer = torch.as_tensor(self.buffer, **i32)
        self.g_stats = torch.zeros(6, **i64)       # oru, reid, ocr, id_recovered, -, slots_full
        self.g_lut = torch.as_tensor(self._group_lut, **i64)
        self.in_cls = torch.zeros(S, N, **i64)
        self.in_active = torch.zeros(S, dtype=torch.bool, device=dev)
        self.in_use_emb = torch.zeros((), dtype=torch.bool, device=dev)
        # pre -> match intermediates (device, static)
        self.g_cost = torch.zeros(S, 5, M, N, dtype=torch.float16, device=dev)
        self.g_det = torch.zeros(S, N, 3, device=dev)
        self.g_twh = torch.zeros(S, M, 4, device=dev)
        self.g_tt = torch.zeros(S, 2, M, M, dtype=torch.float16, device=dev)
        # outputs: device -> pinned ring (one slot per submitted step)
        self.g_outf = torch.zeros(S, M, 6, device=dev)          # box xyxy, score, output mask
        self.g_outi = torch.zeros(S, M, 2, **i64)               # id, cls
        P = c.max_pending
        pin = dict(pin_memory=cuda)
        self.ring_f = torch.zeros(P, S, M, 6, **pin)
        self.ring_i = torch.zeros(P, S, M, 2, dtype=torch.int64, **pin)
        self.ring_act = torch.zeros(P, S, dtype=torch.bool, **pin)
        self.ring_flag = torch.zeros(P, dtype=torch.bool, **pin)
        self.stats_host = torch.zeros(6, dtype=torch.int64, **pin)
        self._pending = []          # (ring slot, active np, events)
        self._events = [[torch.cuda.Event(enable_timing=True) for _ in range(4)] for _ in range(P)] if cuda else None
        self._stats_base = dict(self.stats)

    def _reset_gpu_book(self):
        self.g_state.zero_()
        self.g_ids.fill_(-1)
        for t in (self.g_tsu, self.g_tsg, self.g_hits, self.g_nobs, self.g_birth, self.g_emb_count, self.g_cls,
                  self.g_score, self.g_frame, self.g_stats):
            t.zero_()
        self.g_has_emb.zero_()
        self.g_ref_wh.fill_(1.0)
        self.g_next_id.fill_(1)

    def _gpre(self):
        cost, det, twh, tt = self._gpre_core()
        self.g_cost.copy_(cost)
        self.g_det.copy_(det)
        self.g_twh.copy_(twh)
        self.g_tt.copy_(tt)

    def _gpre_core(self):
        c = self.cfg
        live = (self.g_state != FREE) & self.in_active[:, None]
        lost = (self.g_state == LOST) & live
        kprev = (self.g_nobs - 1).clamp(0, c.ocm_delta).long()
        return self._pre_terms(live, lost, kprev)

    def _gmatch(self):
        self._gmatch_core()

    def _gmatch_core(self):
        """S1..S5 + new tracks + id recovery for every stream at once; writes post_dev + book state."""
        c = self.cfg
        S, M, N = self.S, self.M, self.N
        dev = self.dev
        act = self.in_active
        ue = self.in_use_emb
        st = self.g_state
        live = (st != FREE) & act[:, None]
        frame = self.g_frame + act.long()
        tsu = self.g_tsu + live.int()
        tsg = self.g_tsg + live.int()
        IOU, D2, IOUL, ANG, ED = self.g_cost.float().unbind(1)            # same fp16 values as lapjv path
        dd, dwh = self.g_det[..., 0], self.g_det[..., 1:3]
        twh = self.g_twh[..., :2]
        sc, vd = self.in_scores, self.in_valid
        high = vd & (sc >= c.high_thresh)
        low = vd & (sc >= c.low_thresh) & (sc < c.high_thresh)
        lut = self.g_lut
        if c.class_aware:
            dgrp = lut[self.in_cls.clamp(0, 4095)]
            tgrp = lut[self.g_cls]
        else:
            dgrp = torch.zeros_like(self.in_cls)
            tgrp = torch.zeros_like(self.g_cls)
        same = tgrp[:, :, None] == dgrp[:, None, :]
        hasemb = self.g_has_emb & ue
        conf = live & ((st == TRACKED) | (st == LOST))
        matched = torch.zeros(S, M, dtype=torch.bool, device=dev)
        used = torch.zeros(S, N, dtype=torch.bool, device=dev)
        mdet = torch.zeros(S, M, dtype=torch.long, device=dev)
        stage = torch.zeros(S, M, dtype=torch.int32, device=dev)

        def run(rows, cols, cost, limit, k, matched, used, mdet, stage):
            a = lap_batch(cost, rows & ~matched, cols & ~used, float(limit))
            hit = a >= 0
            ai = a.clamp(min=0)
            hitd = torch.zeros(S, N, dtype=torch.int32, device=dev).scatter_add(1, ai, hit.int()) > 0
            return (matched | hit, used | hitd, torch.where(hit, ai, mdet),
                    torch.where(hit, torch.full_like(stage, k), stage))

        BIGf = float(BIG)
        iou_c = 1.0 - IOU
        # S1: fused IoU / appearance (BoT-SORT) + OCM, Mahalanobis-gated
        emb_c = torch.where(hasemb[..., None], ED, 1.0)
        emb_c = torch.where((emb_c > c.emb_thresh) | (iou_c > c.proximity_thresh), 1.0, emb_c)
        c1 = torch.minimum(iou_c, emb_c)
        has_vel = self.g_nobs >= 2
        c1 = c1 - c.ocm_weight * ANG * sc[:, None, :] * has_vel[..., None]
        c1 = torch.where((D2 > c.gate_chi2) | ~same | ((IOU <= 0) & (emb_c >= 1.0)), BIGf, c1)
        matched, used, mdet, stage = run(conf, high, c1, c.match_thresh, 1, matched, used, mdet, stage)
        # S2: BYTE low-score association, tracks seen last frame only
        c2 = torch.where((D2 > c.gate_chi2) | ~same, BIGf, iou_c)
        matched, used, mdet, stage = run(conf & (tsu == 1), low, c2, c.low_match_thresh, 2, matched, used, mdet, stage)
        # S3: OC-SORT OCR -- last observation vs remaining high dets
        c3 = torch.where(~same, BIGf, 1.0 - IOUL)
        matched, used, mdet, stage = run(conf & (tsu <= c.ocr_max_gap), high, c3, c.ocr_thresh, 3,
                                         matched, used, mdet, stage)
        # S4: long-term re-ID (appearance) for lost tracks (rows empty when no embeddings)
        c4 = torch.where((D2 > c.reid_gate_chi2) | ~same | ~hasemb[..., None], BIGf, ED)
        matched, used, mdet, stage = run(conf & (tsu > 1) & ue, high, c4, c.reid_thresh, 4,
                                         matched, used, mdet, stage)
        # S5: tentative tracks
        c5 = torch.where(~same, BIGf, iou_c)
        matched, used, mdet, stage = run(live & (st == TENTATIVE), high, c5, c.unconfirmed_thresh, 5,
                                         matched, used, mdet, stage)

        # ---- matched tracks (vectorised version of the per-pair loop of the lapjv path) ----
        m = matched
        g2 = lambda x, i: x.gather(1, i)
        g3 = lambda x, i: x.gather(1, i[..., None].expand(S, M, x.shape[-1]))
        sc_b = g2(sc, mdet)
        dwh_b = g3(dwh, mdet)
        ref = self.g_ref_wh
        ratio = torch.minimum((dwh_b / ref.clamp(min=1e-3)).amin(-1), (dwh_b / twh.clamp(min=1e-3)).amin(-1))
        good = m & (sc_b >= c.high_thresh) & (ratio >= c.shape_change_ratio)
        gap = torch.where(good, tsg, 0)
        oru = m & (gap > 1) if c.use_oru else torch.zeros_like(m)
        n_ocr = (m & (stage == 3)).sum()
        n_reid = (m & (stage == 4)).sum()
        n_oru = oru.sum()
        tsu = torch.where(m, 0, tsu)
        hits = self.g_hits + m.int()
        tsg = torch.where(good, 0, tsg)
        nobs = self.g_nobs + good.int()
        ref_wh = torch.where(good[..., None], c.size_ema * ref + (1 - c.size_ema) * dwh_b, ref)
        score = torch.where(m, sc_b, self.g_score)
        cls_b = g2(self.in_cls, mdet)
        cls = torch.where(m & (sc_b >= c.high_thresh), cls_b, self.g_cls)
        confirm = m & (st == TENTATIVE) & (hits >= c.n_init)
        st = torch.where(confirm | (m & (st == LOST)), TRACKED, st)
        eu = ue & good & (g2(dd, mdet) < c.emb_update_max_overlap)
        trust = (sc_b - c.high_thresh) / (1 - c.high_thresh)
        k = self.g_emb_count.double()   # float32 / float64 mix exactly as the NumPy code of the lapjv path
        alpha = torch.minimum((c.emb_alpha + (1 - c.emb_alpha) * (1 - trust)).double(), k / (k + 1.0)).float()
        emb_a = torch.where(eu, alpha, 0.0)
        has_emb = self.g_has_emb | eu
        emb_count = self.g_emb_count + eu.int()
        # ---- unmatched tracks ----
        um = live & ~m
        st = torch.where(um & (st == TENTATIVE), FREE, st)
        st = torch.where(um & (st == TRACKED), LOST, st)
        st = torch.where(um & (st == LOST) & (tsu > self.g_buffer[:, None]), FREE, st)
        # ---- new tracks: k-th leftover det (index order) -> k-th free slot (index order) ----
        rem = vd & (sc >= c.new_track_thresh) & ~used & act[:, None]
        free = (st == FREE) & act[:, None]
        rank_d = rem.long().cumsum(1) - 1
        rank_s = free.long().cumsum(1) - 1
        nrem = rem.sum(1)
        n_full = (nrem - free.sum(1)).clamp(min=0).sum()
        d_of_rank = torch.zeros(S, N + 1, dtype=torch.long, device=dev).scatter(
            1, torch.where(rem, rank_d, N), torch.arange(N, device=dev).expand(S, N))
        new = free & (rank_s < nrem[:, None])
        ndet = torch.where(new, d_of_rank.gather(1, rank_s.clamp(0, N)), 0)
        first = (frame == 1)[:, None]
        st = torch.where(new, torch.where(first, TRACKED, TENTATIVE), st)
        tsu = torch.where(new, 0, tsu)
        tsg = torch.where(new, 0, tsg)
        hits = torch.where(new, 1, hits)
        nobs = torch.where(new, 1, nobs)
        cls = torch.where(new, g2(self.in_cls, ndet), cls)
        score = torch.where(new, g2(sc, ndet), score)
        ref_wh = torch.where(new[..., None], g3(dwh, ndet), ref_wh)
        he = new & ue & (g2(dd, ndet) < c.emb_update_max_overlap)
        has_emb = torch.where(new, he, has_emb)
        emb_count = torch.where(new, he.int(), emb_count)
        birth = torch.where(new, frame[:, None], self.g_birth)
        emb_upd = torch.where(new, he, eu)
        emb_a = torch.where(new, 0.0, emb_a)
        # ids: confirmations first, then first-frame births, each in slot order (as the lapjv path)
        newfirst = new & first
        nconf = confirm.sum(1, keepdim=True)
        rk = torch.where(confirm, confirm.long().cumsum(1) - 1, nconf + newfirst.long().cumsum(1) - 1)
        ids = torch.where(confirm | newfirst, self.g_next_id[:, None] + rk,
                          torch.where(new, -1, self.g_ids))
        next_id = self.g_next_id + nconf[:, 0] + newfirst.sum(1)
        # ---- id recovery: young confirmed tracks vs tracks already lost when they were born ----
        n_rec = torch.zeros((), dtype=torch.long, device=dev)
        if c.id_recovery:
            age = frame[:, None] - birth
            young = ue & (st == TRACKED) & (tsu == 0) & has_emb & (age >= c.young_min_frames) \
                & (age <= c.young_max_frames) & act[:, None]
            lostm = ue & (st == LOST) & has_emb & (tsu > 1) & act[:, None]
            last_seen = frame[:, None] - tsu
            E, G = self.g_tt.float().unbind(1)
            grp = lut[cls]
            ok = (G <= c.reid_gate_chi2) & (last_seen[:, :, None] < birth[:, None, :]) \
                & (grp[:, :, None] == grp[:, None, :])
            a = lap_batch(torch.where(ok, E, BIGf), lostm, young, float(c.reid_thresh))
            hit = a >= 0
            tgt = torch.where(hit, a, M)
            ids = torch.cat([ids, ids.new_zeros(S, 1)], 1).scatter(1, tgt, ids)[:, :M]
            birth = torch.cat([birth, birth.new_zeros(S, 1)], 1).scatter(1, tgt, birth)[:, :M]
            st = torch.where(hit, FREE, st)
            n_rec = hit.sum()
        # ---- write back ----
        self.post_dev.copy_(torch.stack([m.float(), mdet.float(), good.float(), torch.where(oru, gap, 0).float(),
                                         new.float(), ndet.float(), emb_upd.float(), emb_a]))
        self.g_state.copy_(st)
        self.g_ids.copy_(ids)
        self.g_tsu.copy_(tsu)
        self.g_tsg.copy_(tsg)
        self.g_hits.copy_(hits)
        self.g_nobs.copy_(nobs)
        self.g_has_emb.copy_(has_emb)
        self.g_ref_wh.copy_(ref_wh)
        self.g_birth.copy_(birth)
        self.g_emb_count.copy_(emb_count)
        self.g_cls.copy_(cls)
        self.g_score.copy_(score)
        self.g_next_id.copy_(next_id)
        self.g_frame.copy_(frame)
        zero = torch.zeros((), dtype=torch.long, device=dev)
        self.g_stats.add_(torch.stack([n_oru, n_reid, n_ocr, n_rec, zero, n_full]))

    def _gpost(self):
        f, i = self._gpost_core()
        self.g_outf.copy_(f)
        self.g_outi.copy_(i)

    def _gpost_core(self):
        box = self._post_impl(True)    # ORU branch is masked per slot (gap > 1)
        mask = (self.g_state == TRACKED) & (self.g_tsu == 0) & self.in_active[:, None]
        f = torch.cat([box, self.g_score[..., None], mask[..., None].float()], -1)
        return f, torch.stack([self.g_ids, self.g_cls], -1)

    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def submit(self, boxes: torch.Tensor, scores: torch.Tensor, cls: torch.Tensor, valid: torch.Tensor,
               emb: torch.Tensor | None = None, active: np.ndarray | None = None, **_ignored):
        """Enqueue one tracker step on the current CUDA stream without any host sync
        (assign="gpu_hungarian" only).  Results come from :meth:`collect`."""
        assert self.gpu_assign, "submit/collect need assign='gpu_hungarian'"
        cfg, S = self.cfg, self.S
        if len(self._pending) >= cfg.max_pending:
            raise RuntimeError("too many pending steps: call collect()")
        t0 = time.perf_counter()
        k = len(self._pending)
        use_emb = emb is not None and cfg.use_appearance
        active = np.ones(S, bool) if active is None else np.asarray(active, bool)
        n = self._load_inputs(boxes, scores, valid, emb if use_emb else None)
        self.in_cls[:, :n].copy_(cls[:, :n])
        if n < self.N:
            self.in_cls[:, n:] = 0
        # per-step pinned staging slots: an async H2D copy reads them later, so never reuse before collect
        self.ring_act[k] = torch.from_numpy(active)
        self.ring_flag[k] = use_emb
        self.in_active.copy_(self.ring_act[k], non_blocking=True)
        self.in_use_emb.copy_(self.ring_flag[k], non_blocking=True)
        cs = torch.cuda.current_stream(self.dev) if self._events else None
        ev = self._events[k] if self._events else None
        if ev:
            ev[0].record(cs)
        self._run("gpre", self._gpre)
        if ev:
            ev[1].record(cs)
        self._run("gmatch", self._gmatch)
        if ev:
            ev[2].record(cs)
        self._run("gpost", self._gpost)
        if ev:
            ev[3].record(cs)
        self.ring_f[k].copy_(self.g_outf, non_blocking=True)
        self.ring_i[k].copy_(self.g_outi, non_blocking=True)
        self._pending.append((k, active))
        self.timing["host"] = self.timing.get("host", 0.0) + time.perf_counter() - t0

    @torch.no_grad()
    def collect(self) -> list[list[FrameOutput | None]]:
        """One host sync for every pending step; returns their outputs in submission order."""
        t0 = time.perf_counter()
        if not self._pending:
            return []
        self.stats_host.copy_(self.g_stats, non_blocking=True)
        if self.dev.type == "cuda":
            torch.cuda.current_stream(self.dev).synchronize()
        t1 = time.perf_counter()
        res = []
        for k, active in self._pending:
            f = self.ring_f[k].numpy()
            i = self.ring_i[k].numpy()
            outs = []
            for s in range(self.S):
                if not active[s]:
                    outs.append(None)
                    continue
                sl = np.nonzero(f[s, :, 5] > 0.5)[0]
                outs.append(FrameOutput(i[s, sl, 0].copy(), f[s, sl, :4].copy(), f[s, sl, 4].copy(),
                                        i[s, sl, 1].copy()))
            res.append(outs)
            if self._events:
                ev = self._events[k]
                self.timing["gpu_pre"] += ev[0].elapsed_time(ev[1]) / 1e3
                self.timing["match"] += ev[1].elapsed_time(ev[2]) / 1e3
                self.timing["gpu_post"] += ev[2].elapsed_time(ev[3]) / 1e3
        self._pending.clear()
        st = self.stats_host.tolist()
        for j, key in enumerate(("oru", "reid", "ocr", "id_recovered", None, "slots_full")):
            if key:
                self.stats[key] = self._stats_base[key] + st[j]
        t2 = time.perf_counter()
        self.timing["sync_wait"] = self.timing.get("sync_wait", 0.0) + t1 - t0
        self.timing["host"] = self.timing.get("host", 0.0) + t2 - t0
        return res

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
        if self.gpu_assign:
            assert not self._pending, "collect() before reset_stream()"
            self.g_state[s] = FREE
            self.g_ids[s] = -1
            for t in (self.g_tsu, self.g_tsg, self.g_hits, self.g_nobs, self.g_frame):
                t[s] = 0
            self.g_has_emb[s] = False
            self.g_next_id[s] = 1
            self.g_buffer[s] = int(self.buffer[s])

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
        if self.gpu_assign:
            self.submit(boxes, scores, cls, valid, emb, active)
            return self.collect()[0]
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
        if self.dev.type == "cuda":
            torch.cuda.current_stream(self.dev).synchronize()
        self.tsu[live] += 1
        self.tsg[live] += 1
        host = self.cost_host.numpy()
        dh = self.det_host.numpy()
        dd_np, dwh = dh[..., 0], dh[..., 1:3]
        twh = self.twh_host.numpy()
        tt = self.tt_host.numpy()
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
                # partial occlusion = box much smaller than the track's reference size (slow EMA of
                # trusted sizes; follows genuine gradual scale change, not sudden erosion)
                ref = self.ref_wh[s, slot]
                ratio = min((dwh[s, b] / np.maximum(ref, 1e-3)).min(),
                            (dwh[s, b] / np.maximum(twh[s, slot, :2], 1e-3)).min())
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
                    self.ref_wh[s, slot] = cfg.size_ema * ref + (1 - cfg.size_ema) * dwh[s, b]
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
                    k = self.emb_count[s, slot]   # running mean for the first updates, then EMA
                    post[6, s, slot] = 1
                    post[7, s, slot] = min(cfg.emb_alpha + (1 - cfg.emb_alpha) * (1 - trust), k / (k + 1.0))
                    self.has_emb[s, slot] = True
                    self.emb_count[s, slot] = k + 1
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
                self.ref_wh[s, slot] = dwh[s, d]
                self.has_emb[s, slot] = use_emb and dd_np[s, d] < cfg.emb_update_max_overlap
                self.emb_count[s, slot] = int(self.has_emb[s, slot])
                self.birth[s, slot] = self.frame[s]
                if self.has_emb[s, slot]:
                    post[6, s, slot] = 1
                    post[7, s, slot] = 0.0
            # id recovery: young confirmed tracks vs tracks that were already lost when they were born
            if use_emb and cfg.id_recovery:
                age = self.frame[s] - self.birth[s]
                young = np.nonzero((self.state[s] == TRACKED) & (self.tsu[s] == 0) & self.has_emb[s]
                                   & (age >= cfg.young_min_frames) & (age <= cfg.young_max_frames))[0]
                lost = np.nonzero((self.state[s] == LOST) & self.has_emb[s] & (self.tsu[s] > 1))[0]
                if len(young) and len(lost):
                    last_seen = self.frame[s] - self.tsu[s, lost]
                    E = tt[s, 0][np.ix_(lost, young)].astype(np.float32)
                    G = tt[s, 1][np.ix_(lost, young)].astype(np.float32)
                    ok = (G <= cfg.reid_gate_chi2) & (last_seen[:, None] < self.birth[s, young][None, :]) \
                        & (self._group_lut[self.cls[s, lost]][:, None] == self._group_lut[self.cls[s, young]][None, :])
                    r, k = _lap(np.where(ok, E, BIG), cfg.reid_thresh)
                    for a, b in zip(lost[r], young[k]):
                        self.ids[s, b] = self.ids[s, a]
                        self.state[s, a] = FREE
                        self.birth[s, b] = self.birth[s, a]
                        self.stats["id_recovered"] += 1
        t2 = time.perf_counter()

        # ---- GPU post-match update ------------------------------------------------------------
        self.post_dev.copy_(self.post_host, non_blocking=True)
        if any_oru:
            self._run("post_oru", lambda: self._post(True))
        else:
            self._run("post", lambda: self._post(False))
        if self.dev.type == "cuda":
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
        self.timing["match"] += t2 - t1
        self.timing["gpu_post"] += t3 - t2
        return outs
