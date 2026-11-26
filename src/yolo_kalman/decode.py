"""Video sources that yield frames as CUDA uint8 tensors ``[3,H,W]`` (RGB planar).

* :class:`NvDecSource` -- NVDEC hardware decode via PyNvVideoCodec ``ThreadedDecoder`` (background
  decode thread, frames stay in device memory, zero-copy to torch via DLPack).
* :class:`CpuSource`   -- OpenCV decode + upload (fallback, and for synthetic clips/tests).
* :class:`ArraySource` -- in-memory frames (tests).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class NvDecSource:
    def __init__(self, path: str | Path, gpu: int = 0, chunk: int = 8, buffer: int = 32):
        import PyNvVideoCodec as nvc

        self.path = str(path)
        self.gpu = gpu
        self.chunk = chunk
        self.device = torch.device("cuda", gpu)
        # SimpleDecoder only for metadata (cheap; no frames decoded)
        meta = nvc.SimpleDecoder(self.path, gpu_id=gpu, use_device_memory=True).get_stream_metadata()
        self.fps = float(meta.average_fps) or 25.0
        self.num_frames = int(meta.num_frames)
        self.hw = (int(meta.height), int(meta.width))
        self._dec = nvc.ThreadedDecoder(self.path, buffer_size=buffer, gpu_id=gpu, use_device_memory=True,
                                        output_color_type=nvc.OutputColorType.RGBP)
        self.done = False

    def read(self, n: int | None = None) -> torch.Tensor | None:
        """Next up to ``n`` frames as one ``[n,3,H,W]`` uint8 CUDA tensor (copied out of the decoder pool)."""
        if self.done:
            return None
        frames = self._dec.get_batch_frames(n or self.chunk)
        if len(frames) == 0:
            self.done = True
            return None
        with torch.cuda.device(self.device):
            return torch.stack([torch.from_dlpack(f) for f in frames])


class CpuSource:
    def __init__(self, path: str | Path, gpu: int = 0, chunk: int = 8):
        import cv2

        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.hw = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.device = torch.device("cuda", gpu)
        self.chunk = chunk
        self.done = False

    def read(self, n: int | None = None):
        if self.done:
            return None
        out = []
        for _ in range(n or self.chunk):
            ok, f = self.cap.read()
            if not ok:
                self.done = True
                break
            out.append(f[:, :, ::-1].transpose(2, 0, 1))
        if not out:
            return None
        return torch.from_numpy(np.ascontiguousarray(np.stack(out))).to(self.device, non_blocking=True)


class ArraySource:
    def __init__(self, frames: np.ndarray, fps: float = 30.0, gpu: int = 0, chunk: int = 8):
        """frames: [T,H,W,3] uint8 RGB"""
        self.frames = frames
        self.fps = fps
        self.num_frames = len(frames)
        self.hw = frames.shape[1:3]
        self.device = torch.device("cuda", gpu)
        self.chunk = chunk
        self.i = 0
        self.done = False

    def read(self, n: int | None = None):
        if self.i >= self.num_frames:
            self.done = True
            return None
        f = self.frames[self.i:self.i + (n or self.chunk)]
        self.i += len(f)
        return torch.from_numpy(np.ascontiguousarray(f.transpose(0, 3, 1, 2))).to(self.device)


def open_source(path: str | Path, gpu: int = 0, chunk: int = 8, prefer: str = "nvdec"):
    if prefer == "nvdec":
        try:
            return NvDecSource(path, gpu=gpu, chunk=chunk)
        except Exception as e:  # noqa: BLE001
            print(f"[decode] NVDEC failed for {path}: {e!r}; using OpenCV")
    return CpuSource(path, gpu=gpu, chunk=chunk)
