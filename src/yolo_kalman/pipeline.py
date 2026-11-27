"""End-to-end GPU pipeline for S parallel streams on one GPU.

    NVDEC decode (T frames / stream)  ->  GPU letterbox+normalise into one [S*T,3,H,W] fp16 batch
    -> TensorRT YOLO (one batched call) + GPU NMS  ->  un-letterbox  ->  ResNet-18 RoI embeddings
    -> T x MultiStreamTracker.step() (all S streams batched per step)

Images never leave the GPU (only overlay rendering, when enabled, downloads frames).
With ``overlap=True`` three threads overlap, each on its own CUDA stream: a reader (decode + letterbox
of chunk k+2 into one of two input buffers), a detector (TensorRT + embeddings of chunk k+1) and the
tracker (chunk k).
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
        self.S = len(self.paths)
        # Build (and CUDA-graph-capture) the tracker *before* any NVDEC decoder thread exists: decoder
        # threads touching the legacy default stream would invalidate the capture.
        self.tracker = MultiStreamTracker(self.S, 30.0, c.tracker, device=gpu,
                                          emb_dim=self.enc.dim if self.enc else 1)
        self.srcs = [open_source(p, gpu=gpu, chunk=c.chunk, prefer=c.decoder) for p in self.paths]
        for i, src in enumerate(self.srcs):
            self.tracker.reset_stream(i, fps=src.fps)
        self.results = [StreamResult(p, s.fps, s.hw) for p, s in zip(self.paths, self.srcs)]
        H, W = self.det.input_hw
        # two input buffers: the reader thread letterboxes chunk k+1 while TensorRT reads chunk k
        self.inbufs = [torch.empty(self.S * c.chunk, 3, H, W, dtype=torch.float16, device=self.dev)
                       for _ in range(2)]
        self.inbuf = self.inbufs[0]
        self.lb = [letterbox_params(s.hw, (H, W)) for s in self.srcs]
        # un-letterbox constants, built once (a per-chunk torch.tensor(list, device=cuda) is a pageable
        # H2D copy that blocks the host until the stream has finished the TensorRT batch)
        S = self.S
        r = torch.tensor([lb[0] for lb in self.lb], device=self.dev).view(S, 1, 1, 1)
        top = torch.tensor([lb[2][0] for lb in self.lb], device=self.dev, dtype=torch.float32).view(S, 1, 1)
        left = torch.tensor([lb[2][1] for lb in self.lb], device=self.dev, dtype=torch.float32).view(S, 1, 1)
        hw = torch.tensor([s.hw for s in self.srcs], device=self.dev, dtype=torch.float32)
        self._lb_r = r
        self._lb_off = torch.stack([left, top, left, top], -1)
        self._lb_lim = torch.stack([hw[:, 1], hw[:, 0], hw[:, 1], hw[:, 0]], -1).view(S, 1, 1, 4)
        self.same_size = len({tuple(s.hw) for s in self.srcs}) == 1
        self.keep_frames = keep_frames
        self.timing = {"decode": 0.0, "preprocess": 0.0, "detect": 0.0, "embed": 0.0, "track": 0.0}
        self.frames_done = 0
        # tracker kernels are tiny and latency-bound: run them on a high-priority stream so they are
        # scheduled ahead of the big TensorRT batches issued by the producer
        lo, hi = torch.cuda.Stream.priority_range()
        self.prod_stream = torch.cuda.Stream(self.dev, priority=lo)
        self.read_stream = torch.cuda.Stream(self.dev, priority=lo)
        self.track_stream = torch.cuda.Stream(self.dev, priority=hi)
        self.on_chunk = None   # callback(chunk_frames list[S] of uint8 [n,3,H,W] or None, outputs)

    # ---------------------------------------------------------------------------------------
    def _sync(self):
        if self.cfg.profile:
            torch.cuda.current_stream(self.dev).synchronize()

    def _produce(self) -> _Chunk | None:
        part = self._read_prep(0)
        return None if part is None else self._detect_embed(part)

    def _read_prep(self, buf: int, buf_free: torch.cuda.Event | None = None):
        """Decode the next chunk of every stream and letterbox it into ``inbufs[buf]`` (after
        ``buf_free``: TensorRT finished reading that buffer)."""
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
        act = [i for i, f in enumerate(frames) if f is not None]   # finished streams are skipped
        inbuf = self.inbufs[buf]
        if buf_free is not None:
            torch.cuda.current_stream(self.dev).wait_event(buf_free)
        for k, i in enumerate(act):
            letterbox_into(inbuf[k * T:k * T + frames[i].shape[0]], frames[i])
        self._sync()
        t2 = time.perf_counter()
        self.timing["decode"] += t1 - t0
        self.timing["preprocess"] += t2 - t1
        return frames, n, act, buf

    def _detect_embed(self, part) -> _Chunk:
        """Batched detection + un-letterbox + embeddings for a letterboxed chunk."""
        c, S, T = self.cfg, self.S, self.cfg.chunk
        frames, n, act, buf = part
        t2 = time.perf_counter()
        d = self.det(self.inbufs[buf][:len(act) * T])
        K = d.boxes.shape[1]
        if len(act) < S:  # scatter back to the full [S*T] layout
            idx = torch.as_tensor(act, device=self.dev)
            full = lambda x: x.new_zeros(S, T, *x.shape[1:]).index_copy_(0, idx, x.view(len(act), T, *x.shape[1:]))
            d = type(d)(*(full(x).flatten(0, 1) for x in (d.boxes, d.scores, d.cls, d.valid)))
        boxes = d.boxes.view(S, T, K, 4)
        # un-letterbox per stream (vectorised)
        boxes = (boxes - self._lb_off) / self._lb_r
        boxes = torch.minimum(boxes.clamp(min=0), self._lb_lim)
        ch = _Chunk()
        ch.frames, ch.n = frames, n
        ch.boxes = boxes
        ch.scores = d.scores.view(S, T, K)
        ch.cls = d.cls.view(S, T, K)
        wh = boxes[..., 2:] - boxes[..., :2]
        ch.valid = d.valid.view(S, T, K) & ((wh[..., 0] * wh[..., 1]) >= c.tracker.min_box_area)
        tvalid = torch.from_numpy(np.arange(T)[None, :] < n[:, None]).pin_memory()   # truly async H2D
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
        trk = self.tracker
        if trk.gpu_assign:
            # whole chunk enqueued without host syncs (matching runs on the GPU), one collect at the end
            for t in range(T):
                active = ch.n > t
                if not active.any():
                    break
                trk.submit(ch.boxes[:, t], ch.scores[:, t], ch.cls[:, t], ch.valid[:, t],
                           ch.emb[:, t] if ch.emb is not None else None, active=active)
            steps = trk.collect()
        else:
            # one host copy of scores/cls/valid for the whole chunk
            host = torch.stack([ch.scores, ch.cls.float(), ch.valid.float()], -1).cpu().numpy()
            steps = []
            for t in range(T):
                active = ch.n > t
                if not active.any():
                    break
                steps.append(trk.step(
                    ch.boxes[:, t], ch.scores[:, t], ch.cls[:, t], ch.valid[:, t],
                    ch.emb[:, t] if ch.emb is not None else None, active=active,
                    scores_np=host[:, t, :, 0], cls_np=host[:, t, :, 1].astype(np.int64),
                    valid_np=host[:, t, :, 2] > 0.5))
        for outs in steps:
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
            # reader thread (decode + letterbox, double-buffered input) -> detector thread
            # (TensorRT + embeddings) -> this thread (tracker); each on its own CUDA stream
            q: queue.Queue = queue.Queue(maxsize=2)
            q1: queue.Queue = queue.Queue(maxsize=1)
            free: queue.Queue = queue.Queue()
            buf_ev = [None, None]
            for b in range(2):
                free.put(b)
            err = []

            def reader():
                try:
                    torch.cuda.set_device(self.dev)
                    with torch.cuda.stream(self.read_stream), torch.no_grad():
                        while True:
                            b = free.get()
                            part = self._read_prep(b, buf_ev[b])
                            if part is None:
                                q1.put(None)
                                break
                            ev = torch.cuda.Event()
                            ev.record(self.read_stream)
                            q1.put((part, ev))
                except Exception as e:  # noqa: BLE001
                    err.append(e)
                    q1.put(None)

            def producer():
                try:
                    torch.cuda.set_device(self.dev)
                    with torch.cuda.stream(self.prod_stream), torch.no_grad():
                        while True:
                            item = q1.get()
                            if item is None:
                                q.put(None)
                                break
                            part, rev = item
                            self.prod_stream.wait_event(rev)
                            ch = self._detect_embed(part)
                            ev = torch.cuda.Event()
                            ev.record(self.prod_stream)
                            buf_ev[part[3]] = ev
                            free.put(part[3])
                            ch.event = ev
                            q.put(ch)
                except Exception as e:  # noqa: BLE001
                    err.append(e)
                    q.put(None)

            th_r = threading.Thread(target=reader, daemon=True)
            th_r.start()
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
            th_r.join(timeout=10)
            if err:
                raise err[0]
        torch.cuda.synchronize(self.dev)
        self.wall = time.perf_counter() - t0
        self.close()
        return self.results

    def close(self):
        """Release the decoders (and their threads)."""
        for s in self.srcs:
            if hasattr(s, "_dec"):
                s._dec = None
        import gc
        gc.collect()
