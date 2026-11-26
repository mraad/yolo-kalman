"""Detector interface and Ultralytics-YOLO backends (TensorRT FP16 engine, torch FP16 fallback).

Every detector consumes a letterboxed, normalised GPU batch ``[B,3,H,W]`` (float16, 0..1) and
returns padded GPU tensors -- nothing is copied to the host:

    boxes  [B,K,4]  x1y1x2y2 in letterbox pixels
    scores [B,K]
    cls    [B,K]    int64
    valid  [B,K]    bool

To plug in a different (e.g. permissively licensed) detector implement :class:`Detector`.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torchvision


@dataclass
class Detections:
    boxes: torch.Tensor   # [B,K,4] float32 x1y1x2y2 (letterbox coords until unletterboxed)
    scores: torch.Tensor  # [B,K] float32
    cls: torch.Tensor     # [B,K] int64
    valid: torch.Tensor   # [B,K] bool

    def __getitem__(self, i):
        return Detections(self.boxes[i], self.scores[i], self.cls[i], self.valid[i])


class Detector(Protocol):
    input_hw: tuple[int, int]
    max_batch: int
    names: dict[int, str]

    def __call__(self, x: torch.Tensor) -> Detections: ...


def postprocess_raw(raw: torch.Tensor, conf: float, iou: float, max_det: int,
                    classes: torch.Tensor | None) -> Detections:
    """Decode a raw one-to-many YOLO head ``[B, 4+nc, A]`` (xywh + class probs) with GPU NMS.

    One ``batched_nms`` call for the whole batch (offset by image index * nc + class).
    """
    B, C, A = raw.shape
    nc = C - 4
    raw = raw.float()
    xywh = raw[:, :4].transpose(1, 2)              # [B,A,4]
    probs = raw[:, 4:]                             # [B,nc,A]
    if classes is not None:
        mask = torch.zeros(nc, dtype=torch.bool, device=raw.device)
        mask[classes] = True
        probs = probs.masked_fill(~mask[None, :, None], 0)
    score, cls = probs.max(1)                      # [B,A]
    # pre-filter: keep at most 4*max_det candidates per image above conf
    k = min(A, 4 * max_det)
    top_s, top_i = score.topk(k, dim=1)
    top_c = cls.gather(1, top_i)
    top_xywh = xywh.gather(1, top_i[..., None].expand(-1, -1, 4))
    xy1 = top_xywh[..., :2] - top_xywh[..., 2:] / 2
    xy2 = top_xywh[..., :2] + top_xywh[..., 2:] / 2
    top_box = torch.cat([xy1, xy2], -1)
    keep_c = top_s > conf
    bi, ai = keep_c.nonzero(as_tuple=True)
    fb, fs, fc = top_box[bi, ai], top_s[bi, ai], top_c[bi, ai]
    keep = torchvision.ops.batched_nms(fb, fs, bi * nc + fc, iou)  # sorted by score desc
    bi, fb, fs, fc = bi[keep], fb[keep], fs[keep], fc[keep]
    # scatter into padded [B,max_det]: rank within image
    order = torch.argsort(bi, stable=True)  # group by image, preserve score order
    bi, fb, fs, fc = bi[order], fb[order], fs[order], fc[order]
    counts = torch.bincount(bi, minlength=B)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(bi.numel(), device=raw.device) - starts[bi]
    sel = rank < max_det
    bi, rank, fb, fs, fc = bi[sel], rank[sel], fb[sel], fs[sel], fc[sel]
    boxes = torch.zeros(B, max_det, 4, device=raw.device)
    scores = torch.zeros(B, max_det, device=raw.device)
    clsout = torch.zeros(B, max_det, dtype=torch.long, device=raw.device)
    valid = torch.zeros(B, max_det, dtype=torch.bool, device=raw.device)
    boxes[bi, rank] = fb
    scores[bi, rank] = fs
    clsout[bi, rank] = fc
    valid[bi, rank] = True
    return Detections(boxes, scores, clsout, valid)


def read_ultralytics_engine(path: str | Path) -> tuple[bytes, dict]:
    """Ultralytics prefixes engines with ``int32 len + JSON metadata``."""
    data = Path(path).read_bytes()
    meta = {}
    n = struct.unpack("<i", data[:4])[0]
    if 0 < n < 1 << 20:
        try:
            meta = json.loads(data[4:4 + n].decode())
            data = data[4 + n:]
        except (UnicodeDecodeError, json.JSONDecodeError):
            meta = {}
    return data, meta


class TRTYolo:
    """Minimal TensorRT runner: torch tensors in/out, enqueued on the current torch CUDA stream."""

    def __init__(self, engine_path: str | Path, device: int = 0, conf: float = 0.1, iou: float = 0.6,
                 max_det: int = 200, classes: list[int] | None = None):
        import tensorrt as trt

        self.device = torch.device("cuda", device)
        with torch.cuda.device(self.device):
            blob, meta = read_ultralytics_engine(engine_path)
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(blob)
            self.ctx = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        prof = self.engine.get_tensor_profile_shape(self.in_name, 0)  # (min,opt,max)
        self.max_batch = int(prof[2][0])
        side = Path(engine_path).with_suffix(".json")
        info = json.loads(side.read_text()) if side.exists() else {}
        self.input_hw = tuple(info.get("imgsz", meta.get("imgsz", [640, 640])))
        names = info.get("names", meta.get("names", {}))
        self.names = {int(k): v for k, v in names.items()}
        self.in_dtype = torch.float16 if self.engine.get_tensor_dtype(self.in_name) == trt.DataType.HALF else torch.float32
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.classes = torch.tensor(classes, device=self.device) if classes else None

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for s in range(0, x.shape[0], self.max_batch):
            xb = x[s:s + self.max_batch].to(self.in_dtype).contiguous()
            self.ctx.set_input_shape(self.in_name, tuple(xb.shape))
            oshape = tuple(self.ctx.get_tensor_shape(self.out_name))
            out = torch.empty(oshape, dtype=torch.float32, device=self.device)
            self.ctx.set_tensor_address(self.in_name, xb.data_ptr())
            self.ctx.set_tensor_address(self.out_name, out.data_ptr())
            self.ctx.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream)
            outs.append(out)
        return torch.cat(outs) if len(outs) > 1 else outs[0]

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> Detections:
        return postprocess_raw(self.raw(x), self.conf, self.iou, self.max_det, self.classes)


class TorchYolo:
    """Fallback: Ultralytics weights run as a plain FP16 torch module (raw head, our own NMS)."""

    def __init__(self, weights: str | Path, device: int = 0, imgsz=(384, 640), conf: float = 0.1,
                 iou: float = 0.6, max_det: int = 200, classes: list[int] | None = None):
        from ultralytics import YOLO

        self.device = torch.device("cuda", device)
        y = YOLO(str(weights))
        self.names = {int(k): v for k, v in y.names.items()}
        m = y.model.fuse().eval().half().to(self.device)
        head = m.model[-1]
        if getattr(head, "end2end", False):
            head.end2end = False  # use the one-to-many head like the exported engine
        self.model = m
        self.input_hw = tuple(imgsz)
        self.max_batch = 64
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.classes = torch.tensor(classes, device=self.device) if classes else None

    @torch.no_grad()
    def raw(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x.half())
        return out[0] if isinstance(out, (list, tuple)) else out

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> Detections:
        return postprocess_raw(self.raw(x), self.conf, self.iou, self.max_det, self.classes)


def build_detector(spec: str, device: int = 0, **kw) -> Detector:
    """``spec`` is a path to a ``.engine`` (TensorRT) or ``.pt`` (torch FP16) file."""
    if str(spec).endswith(".engine"):
        try:
            return TRTYolo(spec, device=device, **kw)
        except Exception as e:  # noqa: BLE001
            pt = Path(spec).parent / (Path(spec).name.split("_")[0] + ".pt")
            print(f"[detector] TensorRT load failed ({e!r}); falling back to torch FP16 {pt}")
            spec = pt
    return TorchYolo(spec, device=device, **kw)
