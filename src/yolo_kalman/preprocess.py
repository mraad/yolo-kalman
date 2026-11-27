"""GPU letterbox + normalisation and the inverse box mapping."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def letterbox_params(src_hw: tuple[int, int], dst_hw: tuple[int, int]):
    h, w = src_hw
    H, W = dst_hw
    r = min(H / h, W / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (H - nh) // 2, (W - nw) // 2
    return r, (nh, nw), (top, left)


def letterbox_into(out: torch.Tensor, frames: torch.Tensor, pad_value: float = 114 / 255):
    """Resize ``frames`` [N,3,h,w] uint8 (same size) into preallocated ``out`` [N,3,H,W] fp16 in-place.

    Returns (ratio, (top,left)).
    """
    H, W = out.shape[-2:]
    r, (nh, nw), (top, left) = letterbox_params(frames.shape[-2:], (H, W))
    x = frames.to(out.dtype).div_(255)
    if (nh, nw) != tuple(frames.shape[-2:]):
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=False)
    out.fill_(pad_value)
    out[:, :, top:top + nh, left:left + nw] = x
    return r, (top, left)

