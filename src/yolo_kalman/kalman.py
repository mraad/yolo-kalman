"""Batched constant-velocity Kalman filter on (cx, cy, w, h) in PyTorch.

State x = [cx, cy, w, h, vcx, vcy, vw, vh]; measurement z = [cx, cy, w, h].
Noise follows BoT-SORT's xywh filter (std proportional to box w/h) and the NSA Kalman filter
(GIAOTracker): measurement covariance scaled by (1 - detection confidence).

All functions work on arbitrary leading batch dims ``[..., 8]`` / ``[..., 8, 8]`` (e.g. [S, M]).
"""
from __future__ import annotations

import torch

STD_POS = 1.0 / 20
STD_VEL = 1.0 / 160
NDIM = 4


def _wh4(wh: torch.Tensor) -> torch.Tensor:
    """[..., 2] (w,h) -> [..., 4] (w,h,w,h)"""
    return torch.cat([wh, wh], -1)


def motion_matrix(dt: float = 1.0, device=None, dtype=torch.float32) -> torch.Tensor:
    F = torch.eye(8, device=device, dtype=dtype)
    F[:4, 4:] = torch.eye(4, device=device, dtype=dtype) * dt
    return F


def initiate(z: torch.Tensor):
    """z [...,4] -> mean [...,8], cov [...,8,8]"""
    mean = torch.cat([z, torch.zeros_like(z)], -1)
    wh = _wh4(z[..., 2:4])
    std = torch.cat([2 * STD_POS * wh, 10 * STD_VEL * wh], -1)
    cov = torch.diag_embed(std * std)
    return mean, cov


def predict(mean: torch.Tensor, cov: torch.Tensor):
    """x' = F x ; P' = F P F^T + Q  (F is identity + dt block, applied without a matmul for x)."""
    wh = _wh4(mean[..., 2:4])
    std = torch.cat([STD_POS * wh, STD_VEL * wh], -1)
    Q = torch.diag_embed(std * std)
    new_mean = mean.clone()
    new_mean[..., :4] = mean[..., :4] + mean[..., 4:]
    F = motion_matrix(device=mean.device, dtype=mean.dtype)
    new_cov = F @ cov @ F.transpose(-1, -2) + Q
    return new_mean, new_cov


def measurement_cov(mean: torch.Tensor, score: torch.Tensor | None = None,
                    r_scale: torch.Tensor | None = None) -> torch.Tensor:
    """R [...,4,4]; NSA: R *= (1 - score); optional extra per-measurement scale (low-quality obs)."""
    wh = _wh4(mean[..., 2:4])
    var = (STD_POS * wh) ** 2
    if score is not None:
        var = var * (1.0 - score.clamp(0.0, 0.99))[..., None]
    if r_scale is not None:
        var = var * r_scale[..., None]
    return torch.diag_embed(var)


def project(mean: torch.Tensor, cov: torch.Tensor, score: torch.Tensor | None = None,
            r_scale: torch.Tensor | None = None):
    """Innovation distribution: H x, H P H^T + R."""
    return mean[..., :4], cov[..., :4, :4] + measurement_cov(mean, score, r_scale)


def update(mean: torch.Tensor, cov: torch.Tensor, z: torch.Tensor, score: torch.Tensor | None = None,
           r_scale: torch.Tensor | None = None):
    pm, S = project(mean, cov, score, r_scale)
    PHt = cov[..., :, :4]                                # P H^T  [...,8,4]
    L = torch.linalg.cholesky(S)
    # K = P H^T S^-1  ->  solve S K^T = H P
    Kt = torch.cholesky_solve(PHt.transpose(-1, -2), L)  # [...,4,8]
    K = Kt.transpose(-1, -2)
    innov = z - pm
    new_mean = mean + (K @ innov[..., None])[..., 0]
    new_cov = cov - K @ S @ Kt
    new_cov = 0.5 * (new_cov + new_cov.transpose(-1, -2))
    return new_mean, new_cov


def mahalanobis_sq(mean: torch.Tensor, cov: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Squared Mahalanobis distance of measurements to tracks.

    mean [...,M,8], cov [...,M,8,8], z [...,N,4] -> d2 [...,M,N]
    """
    pm, S = project(mean, cov)
    L = torch.linalg.cholesky(S)                         # [...,M,4,4]
    d = z.unsqueeze(-3) - pm.unsqueeze(-2)               # [...,M,N,4]
    # solve L y = d^T  for all N at once: [...,M,4,N]
    y = torch.linalg.solve_triangular(L, d.transpose(-1, -2), upper=False)
    return (y * y).sum(-2)


# ---- box helpers -------------------------------------------------------------------------
def xyxy_to_xywh(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([(b[..., :2] + b[..., 2:]) / 2, b[..., 2:] - b[..., :2]], -1)


def xywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], -1)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [...,M,4], b [...,N,4] xyxy -> [...,M,N]"""
    a_ = a.unsqueeze(-2)
    b_ = b.unsqueeze(-3)
    lt = torch.maximum(a_[..., :2], b_[..., :2])
    rb = torch.minimum(a_[..., 2:], b_[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
    return inter / (area_a.unsqueeze(-1) + area_b.unsqueeze(-2) - inter + 1e-9)
