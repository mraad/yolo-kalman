"""Appearance embeddings for re-identification after long occlusions.

Choice (see README): torchvision ResNet-18 ImageNet trunk (global-average-pooled layer4, 512-d,
L2-normalised) on RoI-aligned crops taken straight from the decoded CUDA frames.

Why not OSNet/torchreid: OSNet re-ID weights are person-only (Market/MSMT) and distributed via
Google Drive; torchreid is unmaintained on PyPI.  We track people *and* vehicles, so a generic,
class-agnostic appearance descriptor that installs from the PyTorch index is the pragmatic pick.
The encoder is behind a tiny interface (``__call__(frames, boxes, batch_idx) -> [n,D]``) so a
proper re-ID net can replace it.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torchvision.ops import roi_align

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class AppearanceEncoder:
    def __init__(self, device: int = 0, crop_hw=(128, 64), arch: str = "resnet18"):
        self.device = torch.device("cuda", device)
        if arch == "resnet18":
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.dim = 512
        else:
            raise ValueError(arch)
        m.fc = torch.nn.Identity()
        self.model = m.eval().half().to(self.device).to(memory_format=torch.channels_last)
        self.crop_hw = crop_hw
        self.mean = _MEAN.to(self.device).half()
        self.std = _STD.to(self.device).half()

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor, boxes: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """frames [B,3,H,W] uint8 CUDA; boxes [n,4] xyxy source px; batch_idx [n] -> [n,D] unit vectors."""
        if boxes.numel() == 0:
            return torch.zeros(0, self.dim, device=self.device)
        rois = torch.cat([batch_idx[:, None].to(boxes.dtype), boxes], 1)
        # roi_align on uint8 is unsupported: convert lazily per frame batch in fp16
        crops = roi_align(frames.half(), rois.half(), self.crop_hw, spatial_scale=1.0, sampling_ratio=2,
                          aligned=True)
        crops = ((crops / 255) - self.mean) / self.std
        f = self.model(crops.contiguous(memory_format=torch.channels_last))
        return F.normalize(f.float(), dim=1)
