"""Batched constant-velocity Kalman filter on (cx, cy, w, h) in PyTorch.

State x = [cx, cy, w, h, vcx, vcy, vw, vh]; measurement z = [cx, cy, w, h].
Noise follows BoT-SORT's xywh filter (std proportional to box w/h) and the NSA Kalman filter
(GIAOTracker): measurement covariance scaled by (1 - detection confidence).

All functions work on arbitrary leading batch dims ``[..., 8]`` / ``[..., 8, 8]`` (e.g. [S, M]).
"""
from __future__ import annotations

import torch

# When True, 4x4 Cholesky factorisations / solves are written out element-wise (fusable by
# torch.compile/inductor into a handful of kernels instead of cuSOLVER/cuBLAS batched calls).
UNROLLED_4X4 = False

STD_POS = 1.0 / 20
STD_VEL = 1.0 / 160
NDIM = 4


def _wh4(wh: torch.Tensor) -> torch.Tensor:
    """[..., 2] (w,h) -> [..., 4] (w,h,w,h)"""
    return torch.cat([wh, wh], -1)


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
    new_mean = torch.cat([mean[..., :4] + mean[..., 4:], mean[..., 4:]], -1)
    return new_mean, _fpf(cov, None) + Q


def _fpf(cov: torch.Tensor, d: torch.Tensor | None) -> torch.Tensor:
    """F P F^T for F = [[I, d I], [0, I]] written block-wise (element-wise ops only: no batched
    8x8 matmuls, so inductor fuses it).  d: None (= 1) or [..., 1]."""
    A, B = cov[..., :4, :4], cov[..., :4, 4:]
    C, D = cov[..., 4:, :4], cov[..., 4:, 4:]
    if d is None:
        dB, dC, dD, ddD = B, C, D, D
    else:
        dd = d[..., None]
        dB, dC, dD = dd * B, dd * C, dd * D
        ddD = dd * dD
    top = torch.cat([A + dB + dC + ddD, B + dD], -1)
    bot = torch.cat([C + dD, D], -1)
    return torch.cat([top, bot], -2)


def predict_dt(mean: torch.Tensor, cov: torch.Tensor, dt: torch.Tensor):
    """Exact ``dt``-step prediction (Q held at the starting box size): dt [...] (float, >= 0).

    F^dt = [[I, dt I], [0, I]];  sum_j F^j Q F^jT = [[dt qp + qv S2, qv S1], [qv S1, dt qv]]
    with S1 = sum j = dt(dt-1)/2, S2 = sum j^2 = (dt-1)dt(2dt-1)/6.  dt=1 equals :func:`predict`.
    """
    wh = _wh4(mean[..., 2:4])
    qp, qv = (STD_POS * wh) ** 2, (STD_VEL * wh) ** 2
    d = dt[..., None]
    s1 = d * (d - 1) / 2
    s2 = (d - 1) * d * (2 * d - 1) / 6
    new_mean = torch.cat([mean[..., :4] + mean[..., 4:] * d, mean[..., 4:]], -1)
    q11, q12, q22 = (torch.diag_embed(x) for x in (d * qp + qv * s2, qv * s1, d * qv))
    Qs = torch.cat([torch.cat([q11, q12], -1), torch.cat([q12, q22], -1)], -2)
    return new_mean, _fpf(cov, d) + Qs


def measurement_cov(mean: torch.Tensor, score: torch.Tensor | None = None,
                    r_scale: torch.Tensor | None = None) -> torch.Tensor:
    """R [...,4,4]; NSA: R *= (1 - score); optional extra per-measurement scale (low-quality obs)."""
    wh = _wh4(mean[..., 2:4])
    var = (STD_POS * wh) ** 2
    if score is not None:
        var = var * (1.0 - score.clamp(0.0, 0.99))[..., None]
    if r_scale is not None:  # scalar per measurement [...] or per dimension [...,4]
        var = var * (r_scale if r_scale.dim() == var.dim() else r_scale[..., None])
    return torch.diag_embed(var)


def project(mean: torch.Tensor, cov: torch.Tensor, score: torch.Tensor | None = None,
            r_scale: torch.Tensor | None = None):
    """Innovation distribution: H x, H P H^T + R."""
    return mean[..., :4], cov[..., :4, :4] + measurement_cov(mean, score, r_scale)


def _chol4(S: torch.Tensor):
    """Element-wise Cholesky of SPD [...,4,4] -> list-of-lists L[i][j] (j<=i) of [...] tensors."""
    L = [[None] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i + 1):
            acc = S[..., i, j]
            for k in range(j):
                acc = acc - L[i][k] * L[j][k]
            if i == j:
                L[i][j] = torch.sqrt(acc.clamp(min=1e-12))
            else:
                L[i][j] = acc / L[j][j]
    return L


def _fwd4(L, b):
    """Solve L y = b; b list of 4 tensors [...]."""
    y = []
    for i in range(4):
        acc = b[i]
        for k in range(i):
            acc = acc - L[i][k] * y[k]
        y.append(acc / L[i][i])
    return y


def _bwd4(L, y):
    """Solve L^T x = y."""
    x = [None] * 4
    for i in reversed(range(4)):
        acc = y[i]
        for k in range(i + 1, 4):
            acc = acc - L[k][i] * x[k]
        x[i] = acc / L[i][i]
    return x


def spd_solve(S: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """S^-1 B for SPD S [...,4,4], B [...,4,k]."""
    if not UNROLLED_4X4:
        return torch.cholesky_solve(B, torch.linalg.cholesky_ex(S)[0])
    L = _chol4(S)
    Lb = [[t.unsqueeze(-1) for t in row if t is not None] for row in L]
    Lf = [[Lb[i][j] if j <= i else None for j in range(4)] for i in range(4)]
    x = _bwd4(Lf, _fwd4(Lf, [B[..., i, :] for i in range(4)]))
    return torch.stack(x, -2)


def tri_solve_sq_norm(S: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """sum((L^-1 d)^2) for S = L L^T [...,4,4] and D [...,4,N] -> [...,N]."""
    if not UNROLLED_4X4:
        L = torch.linalg.cholesky_ex(S)[0]
        y = torch.linalg.solve_triangular(L, D, upper=False)
        return (y * y).sum(-2)
    L = _chol4(S)
    Lf = [[L[i][j].unsqueeze(-1) if j <= i else None for j in range(4)] for i in range(4)]
    y = _fwd4(Lf, [D[..., i, :] for i in range(4)])
    return y[0] ** 2 + y[1] ** 2 + y[2] ** 2 + y[3] ** 2


def update(mean: torch.Tensor, cov: torch.Tensor, z: torch.Tensor, score: torch.Tensor | None = None,
           r_scale: torch.Tensor | None = None):
    pm, S = project(mean, cov, score, r_scale)
    PHt = cov[..., :, :4]                                # P H^T  [...,8,4]
    # K = P H^T S^-1  ->  solve S K^T = H P   (cholesky_ex: no host sync -> CUDA-graph safe)
    Kt = spd_solve(S, PHt.transpose(-1, -2))             # [...,4,8]
    K = Kt.transpose(-1, -2)
    innov = z - pm
    # element-wise products + small reductions instead of batched matmuls (fusable by inductor);
    # K S K^T = P H^T S^-1 H P = PHt Kt
    new_mean = mean + (K * innov[..., None, :]).sum(-1)
    new_cov = cov - (PHt[..., :, :, None] * Kt[..., None, :, :]).sum(-2)
    new_cov = 0.5 * (new_cov + new_cov.transpose(-1, -2))
    return new_mean, new_cov


def mahalanobis_sq(mean: torch.Tensor, cov: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Squared Mahalanobis distance of measurements to tracks.

    mean [...,M,8], cov [...,M,8,8], z [...,N,4] -> d2 [...,M,N]
    """
    pm, S = project(mean, cov)
    d = z.unsqueeze(-3) - pm.unsqueeze(-2)               # [...,M,N,4]
    return tri_solve_sq_norm(S, d.transpose(-1, -2))     # solve L y = d^T for all N at once


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
