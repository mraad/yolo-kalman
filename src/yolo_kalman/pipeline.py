"""End-to-end GPU pipeline for S parallel streams on one GPU.

    NVDEC decode (T frames / stream)  ->  GPU letterbox+normalise into one [S*T,3,H,W] fp16 batch
    -> TensorRT YOLO (one batched call) + GPU NMS  ->  un-letterbox  ->  ResNet-18 RoI embeddings
    -> T x MultiStreamTracker.step() (all S streams batched per step)

Images never leave the GPU (only overlay rendering, when enabled, downloads frames).
With ``overlap=True`` decode+detect+embed of chunk k+1 runs in a producer thread on its own CUDA
stream while the tracker consumes chunk k.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .decode import open_source
from .detector import build_detector
from .preprocess import letterbox_into, letterbox_params
from .reid import AppearanceEncoder
from .tracker import MultiStreamTracker, TrackerConfig


@dataclass
class PipelineConfig:
    engine: str = "models/yolo26m_384x640_b64_e2e_fp16.engine"
    chunk: int = 8                 # frames per stream per detector batch
    conf: float = 0.1              # detector threshold (BYTE needs low-score boxes)
    nms_iou: float = 0.6
    max_det: int = 150
    classes: list[int] | None = None
    use_reid: bool = True
    reid_min_score: float = 0.5    # embed only detections used by re-ID stages
    decoder: str = "nvdec"
    overlap: bool = True
    max_frames: int | None = None  # per stream
    profile: bool = False          # synchronise between stages for an exact time split
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


@dataclass
class StreamResult:
    path: str
    fps: float
    hw: tuple[int, int]
    frames: int = 0
    frames_read: int = 0
    tracks: list = field(default_factory=list)   # per frame: (ids[n], boxes[n,4], scores[n], cls[n])


class _Chunk:
    __slots__ = ("frames", "n", "boxes", "scores", "cls", "valid", "emb", "host", "event")


class Pipeline:
    def __init__(self, paths: list[str], gpu: int = 0, cfg: PipelineConfig | None = None, keep_frames=False):
        self.cfg = cfg or PipelineConfig()
        self.gpu = gpu
        self.dev = torch.device("cuda", gpu)
        torch.cuda.set_device(self.dev)
        c = self.cfg
        self.det = build_detector(c.engine, device=gpu, conf=c.conf, iou=c.nms_iou, max_det=c.max_det,
                                  classes=c.classes)
        self.enc = AppearanceEncoder(device=gpu) if c.use_reid else None
        self.paths = list(paths)
        self.srcs = [open_source(p, gpu=gpu, chunk=c.chunk, prefer=c.decoder) for p in self.paths]
        self.S = len(self.srcs)
        self.results = [StreamResult(p, s.fps, s.hw) for p, s in zip(self.paths, self.srcs)]
        self.tracker = MultiStreamTracker(self.S, [s.fps for s in self.srcs], c.tracker, device=gpu,
                                          emb_dim=self.enc.dim if self.enc else 1)
        H, W = self.det.input_hw
        self.inbuf = torch.empty(self.S * c.chunk, 3, H, W, dtype=torch.float16, device=self.dev)
        self.lb = [letterbox_params(s.hw, (H, W)) for s in self.srcs]
        self.same_size = len({tuple(s.hw) for s in self.srcs}) == 1
        self.keep_frames = keep_frames
        self.timing = {"decode": 0.0, "preprocess": 0.0, "detect": 0.0, "embed": 0.0, "track": 0.0}
        self.frames_done = 0
        # tracker kernels are tiny and latency-bound: run them on a high-priority stream so they are
        # scheduled ahead of the big TensorRT batches issued by the producer
        lo, hi = torch.cuda.Stream.priority_range()
        self.prod_stream = torch.cuda.Stream(self.dev, priority=lo)
        self.track_stream = torch.cuda.Stream(self.dev, priority=hi)
        self.on_chunk = None   # callback(chunk_frames list[S] of uint8 [n,3,H,W] or None, outputs)

    # ---------------------------------------------------------------------------------------
    def _sync(self):
        if self.cfg.profile:
            torch.cuda.current_stream(self.dev).synchronize()

    def _produce(self) -> _Chunk | None:
        c, S, T = self.cfg, self.S, self.cfg.chunk
        t0 = time.perf_counter()
        frames = []
        for i, src in enumerate(self.srcs):
            done = self.results[i].frames_read
            want = T
            if c.max_frames is not None:
                want = min(T, c.max_frames - done)
            f = src.read(want) if want > 0 else None
            frames.append(f)
            self.results[i].frames_read = done + (0 if f is None else f.shape[0])
        if all(f is None for f in frames):
            return None
        self._sync()
        t1 = time.perf_counter()
        n = np.array([0 if f is None else f.shape[0] for f in frames])
        for i, f in enumerate(frames):
            if f is not None:
                letterbox_into(self.inbuf[i * T:i * T + f.shape[0]], f)
        self._sync()
        t2 = time.perf_counter()
        d = self.det(self.inbuf)
        K = d.boxes.shape[1]
        boxes = d.boxes.view(S, T, K, 4)
        # un-letterbox per stream (vectorised)
        r = torch.tensor([lb[0] for lb in self.lb], device=self.dev).view(S, 1, 1, 1)
        top = torch.tensor([lb[2][0] for lb in self.lb], device=self.dev, dtype=torch.float32).view(S, 1, 1)
        left = torch.tensor([lb[2][1] for lb in self.lb], device=self.dev, dtype=torch.float32).view(S, 1, 1)
        hw = torch.tensor([s.hw for s in self.srcs], device=self.dev, dtype=torch.float32)
        off = torch.stack([left, top, left, top], -1)
        boxes = (boxes - off) / r
        lim = torch.stack([hw[:, 1], hw[:, 0], hw[:, 1], hw[:, 0]], -1).view(S, 1, 1, 4)
        boxes = torch.minimum(boxes.clamp(min=0), lim)
        ch = _Chunk()
        ch.frames, ch.n = frames, n
        ch.boxes = boxes
        ch.scores = d.scores.view(S, T, K)
        ch.cls = d.cls.view(S, T, K)
        wh = boxes[..., 2:] - boxes[..., :2]
        ch.valid = d.valid.view(S, T, K) & ((wh[..., 0] * wh[..., 1]) >= c.tracker.min_box_area)
        tvalid = torch.zeros(S, T, dtype=torch.bool)
        for i in range(S):
            tvalid[i, :n[i]] = True
        ch.valid &= tvalid.to(self.dev, non_blocking=True)[..., None]
        self._sync()
        t3 = time.perf_counter()
        ch.emb = None
        if self.enc is not None:
            ch.emb = torch.zeros(S, T, K, self.enc.dim, device=self.dev)
            sel = ch.valid & (ch.scores >= c.reid_min_score)          # [S,T,K]
            if self.same_size:
                # one RoI-align + ResNet call for every stream: frames stacked as [S*T,...]
                allf = torch.cat([f if f.shape[0] == T else
                                  torch.cat([f, f.new_zeros(T - f.shape[0], *f.shape[1:])])
                                  for f in frames if f is not None]) if all(f is not None for f in frames) else None
                if allf is not None:
                    si, ti, ki = sel.nonzero(as_tuple=True)
                    e = self.enc(allf, boxes[si, ti, ki], si * T + ti)
                    ch.emb[si, ti, ki] = e
                else:
                    self._embed_per_stream(ch, frames, sel, boxes, n)
            else:
                self._embed_per_stream(ch, frames, sel, boxes, n)
        self._sync()
        t4 = time.perf_counter()
        self.timing["decode"] += t1 - t0
        self.timing["preprocess"] += t2 - t1
        self.timing["detect"] += t3 - t2
        self.timing["embed"] += t4 - t3
        return ch

    def _embed_per_stream(self, ch, frames, sel, boxes, n):
        for i, f in enumerate(frames):
            if f is None:
                continue
            ti, ki = sel[i, :n[i]].nonzero(as_tuple=True)
            ch.emb[i, ti, ki] = self.enc(f, boxes[i, ti, ki], ti)

    def _consume(self, ch: _Chunk):
        t0 = time.perf_counter()
        T = self.cfg.chunk
        # one host copy of scores/cls/valid for the whole chunk
        host = torch.stack([ch.scores, ch.cls.float(), ch.valid.float()], -1).cpu().numpy()
        for t in range(T):
            active = ch.n > t
            if not active.any():
                break
            outs = self.tracker.step(
                ch.boxes[:, t], ch.scores[:, t], ch.cls[:, t], ch.valid[:, t],
                ch.emb[:, t] if ch.emb is not None else None, active=active,
                scores_np=host[:, t, :, 0], cls_np=host[:, t, :, 1].astype(np.int64),
                valid_np=host[:, t, :, 2] > 0.5)
            for i, o in enumerate(outs):
                if o is not None:
                    self.results[i].tracks.append((o.ids, o.boxes, o.scores, o.cls))
                    self.results[i].frames += 1
        self.frames_done += int(ch.n.sum())
        if self.on_chunk is not None:
            self.on_chunk(ch.frames, [r for r in self.results], ch.n)
        torch.cuda.current_stream(self.dev).synchronize()  # chunk tensors may be recycled by the producer
        self.timing["track"] += time.perf_counter() - t0

    # ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def run(self):
        t0 = time.perf_counter()
        if not self.cfg.overlap:
            while (ch := self._produce()) is not None:
                self._consume(ch)
        else:
            q: queue.Queue = queue.Queue(maxsize=2)
            err = []

            def producer():
                try:
                    torch.cuda.set_device(self.dev)
                    with torch.cuda.stream(self.prod_stream), torch.no_grad():
                        while True:
                            ch = self._produce()
                            if ch is not None:
                                ev = torch.cuda.Event()
                                ev.record(self.prod_stream)
                                ch.event = ev
                            q.put(ch)
                            if ch is None:
                                break
                except Exception as e:  # noqa: BLE001
                    err.append(e)
                    q.put(None)

            th = threading.Thread(target=producer, daemon=True)
            th.start()
            with torch.cuda.stream(self.track_stream):
                while True:
                    ch = q.get()
                    if ch is None:
                        break
                    self.track_stream.wait_event(ch.event)
                    self._consume(ch)
            th.join()
            if err:
                raise err[0]
        torch.cuda.synchronize(self.dev)
        self.wall = time.perf_counter() - t0
        return self.results
