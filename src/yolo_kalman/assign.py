"""Batched exact linear assignment on the GPU (Hungarian / Kuhn-Munkres, shortest augmenting path).

Problem (per batch element b, identical to ``lap.lapjv(c, extend_cost=True, cost_limit=limit)``):
rows ``r`` with ``row_mask[b, r]``, columns ``k`` with ``col_mask[b, k]``; every row and column may stay
unmatched at cost ``limit / 2``.  lap builds the (R+C)x(R+C) extended matrix for this; its optimum is

    min  sum_{matched (r,k)} (c_rk - limit)   +   limit/2 * (R + C)

so a pair with ``c_rk > limit`` is never part of an optimum (forbidden here), a pair with
``c_rk == limit`` ties with "unmatched" (we match it: real columns come before dummies, ties go to the
lowest column index).  We solve the equivalent rectangular problem n x (m + n): the n compacted rows
x (m compacted real columns with cost c - limit, then n dummy columns with cost 0 = "row unmatched").
Rows / columns with no allowed pair are dropped before solving (they are unmatched in every optimum).

Algorithm: the O(n^2 (m+n)) Hungarian method in its shortest-augmenting-path form with row/column
dual potentials (Kuhn-Munkres as formulated by Jonker-Volgenant / the classical "e-maxx" variant):
rows are inserted one at a time; for each, a Dijkstra search over reduced costs
``c_ij - u_i - v_j >= 0`` finds the shortest augmenting path to a free column, duals are updated
by the path length, and the path is flipped.  Every step keeps complementary slackness, so the
result is an optimal assignment (exact; float64 duals, float32 costs).  Not an approximation: no
epsilon, no iteration cap.

CUDA kernel: one warp per batch element (= stream); columns are strided over the 32 lanes, the
Dijkstra arg-min is a warp-shuffle reduction (ties -> lowest column index, deterministic).  All
data-dependent loops live inside the kernel, so a call is one fixed launch: CUDA-graph capturable,
no host sync.  Exposed as the torch custom op ``yolo_kalman::lap_batch`` (works under
torch.compile).  ``lap_batch_torch`` is a pure-PyTorch batched implementation of the same
algorithm (needs a host sync per Dijkstra step); it is the CPU fallback and the measured
alternative (see README).
"""
from __future__ import annotations

import os
import sys
import threading

import numpy as np
import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define FULL 0xffffffffu

// one warp (blockDim.x == 32) per batch element
__global__ void lap_sap_kernel(const float* __restrict__ cost, const bool* __restrict__ rmask,
                               const bool* __restrict__ cmask, int64_t* __restrict__ out,
                               int R, int C, double limit) {
  extern __shared__ unsigned char smem[];
  const int b = blockIdx.x, lane = threadIdx.x;
  const float* cb = cost + (size_t)b * R * C;
  const bool* rm = rmask + (size_t)b * R;
  const bool* cm = cmask + (size_t)b * C;
  int64_t* ob = out + (size_t)b * R;
  const int WMAX = R + C;
  // shared layout (8-byte arrays first)
  double* u    = (double*)smem;                 // [R+1]   row duals (1-based)
  double* v    = u + (R + 1);                   // [W+1]   column duals (1-based, 0 = virtual)
  double* minv = v + (WMAX + 1);                // [W+1]
  int* p       = (int*)(minv + (WMAX + 1));     // [W+1]   row assigned to column (0 = free)
  int* way     = p + (WMAX + 1);                // [W+1]   predecessor column on the path
  int* ri      = way + (WMAX + 1);              // [R]     compacted row -> original row
  int* cj      = ri + R;                        // [C]     compacted col -> original col
  unsigned char* used = (unsigned char*)(cj + C);  // [W+1]

  for (int r = lane; r < R; r += 32) ob[r] = -1;
  const unsigned lt = (1u << lane) - 1u;
  // ---- compaction (order preserving): rows / cols that have at least one allowed pair ----
  int n = 0;
  for (int base = 0; base < R; base += 32) {
    int r = base + lane;
    bool f = false;
    if (r < R && rm[r]) {
      const float* row = cb + (size_t)r * C;
      for (int k = 0; k < C; ++k) {
        float c = row[k];
        if (cm[k] && (double)c <= limit) { f = true; break; }
      }
    }
    unsigned bal = __ballot_sync(FULL, f);
    if (f) ri[n + __popc(bal & lt)] = r;
    n += __popc(bal);
  }
  int m = 0;
  for (int base = 0; base < C; base += 32) {
    int k = base + lane;
    bool f = false;
    if (k < C && cm[k]) {
      for (int r = 0; r < R; ++r) {
        if (rm[r] && (double)cb[(size_t)r * C + k] <= limit) { f = true; break; }
      }
    }
    unsigned bal = __ballot_sync(FULL, f);
    if (f) cj[m + __popc(bal & lt)] = k;
    m += __popc(bal);
  }
  __syncwarp();
  if (n == 0 || m == 0) return;
  const int W = m + n;   // real columns 1..m, dummy columns m+1..m+n (cost 0 = row unmatched)
  for (int i = lane; i <= n; i += 32) u[i] = 0.0;
  for (int j = lane; j <= W; j += 32) { v[j] = 0.0; p[j] = 0; way[j] = 0; }
  __syncwarp();
  const double INF = CUDART_INF;
  for (int i = 1; i <= n; ++i) {
    for (int j = lane; j <= W; j += 32) { minv[j] = INF; used[j] = 0; }
    if (lane == 0) p[0] = i;
    __syncwarp();
    int j0 = 0;
    while (true) {
      if (lane == 0) used[j0] = 1;
      __syncwarp();
      const int i0 = p[j0];
      const double ui0 = u[i0];
      const float* row = cb + (size_t)ri[i0 - 1] * C;
      double delta = INF;
      int j1 = 0x7fffffff;
      for (int j = 1 + lane; j <= W; j += 32) {
        if (!used[j]) {
          double a;
          if (j <= m) {
            float c = row[cj[j - 1]];
            a = ((double)c <= limit) ? (double)c - limit : INF;
          } else {
            a = 0.0;
          }
          double cur = a - ui0 - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
      }
      // warp arg-min, ties -> lowest column index
      for (int off = 16; off > 0; off >>= 1) {
        double d2 = __shfl_xor_sync(FULL, delta, off);
        int k2 = __shfl_xor_sync(FULL, j1, off);
        if (d2 < delta || (d2 == delta && k2 < j1)) { delta = d2; j1 = k2; }
      }
      for (int j = lane; j <= W; j += 32) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else minv[j] -= delta;
      }
      __syncwarp();
      j0 = j1;
      if (p[j0] == 0) break;
    }
    if (lane == 0) {
      do { int jp = way[j0]; p[j0] = p[jp]; j0 = jp; } while (j0);
    }
    __syncwarp();
  }
  for (int j = 1 + lane; j <= m; j += 32) {
    int i = p[j];
    if (i) ob[ri[i - 1]] = cj[j - 1];
  }
}

torch::Tensor lap_batch_cuda(torch::Tensor cost, torch::Tensor rmask, torch::Tensor cmask, double limit) {
  TORCH_CHECK(cost.is_cuda() && cost.dim() == 3 && cost.scalar_type() == torch::kFloat32);
  TORCH_CHECK(rmask.scalar_type() == torch::kBool && cmask.scalar_type() == torch::kBool);
  auto c = cost.contiguous();
  auto rm = rmask.contiguous();
  auto cmk = cmask.contiguous();
  const int B = c.size(0), R = c.size(1), C = c.size(2);
  auto out = torch::empty({B, R}, c.options().dtype(torch::kInt64));
  if (B == 0 || R == 0) return out;
  if (C == 0) { out.fill_(-1); return out; }
  const int W = R + C;
  size_t sh = (size_t)(R + 1) * 8 + (size_t)(W + 1) * 16 + (size_t)(W + 1) * 8 + (size_t)(R + C) * 4 + (W + 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (sh > 48 * 1024)
    cudaFuncSetAttribute(lap_sap_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
  lap_sap_kernel<<<B, 32, sh, stream>>>(c.data_ptr<float>(), rm.data_ptr<bool>(), cmk.data_ptr<bool>(),
                                        out.data_ptr<int64_t>(), R, C, limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

_CPP_SRC = "torch::Tensor lap_batch_cuda(torch::Tensor cost, torch::Tensor rmask, torch::Tensor cmask, double limit);"

_ext = None
_ext_lock = threading.Lock()


def _load_ext():
    """JIT-build the CUDA extension (cached under ~/.cache/torch_extensions)."""
    global _ext
    if _ext is None:
        with _ext_lock:
            if _ext is None:
                from torch.utils.cpp_extension import load_inline

                if "TORCH_CUDA_ARCH_LIST" not in os.environ:
                    maj, mnr = torch.cuda.get_device_capability()
                    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{maj}.{mnr}"
                os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
                vbin = os.path.dirname(sys.executable)   # ninja lives in the venv's bin
                if vbin not in os.environ.get("PATH", "").split(os.pathsep):
                    os.environ["PATH"] = vbin + os.pathsep + os.environ.get("PATH", "")
                _ext = load_inline("yolo_kalman_lap", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
                                   functions=["lap_batch_cuda"], extra_cuda_cflags=["-O3"],
                                   verbose=False)
    return _ext


@torch.library.custom_op("yolo_kalman::lap_batch", mutates_args=())
def lap_batch(cost: torch.Tensor, row_mask: torch.Tensor, col_mask: torch.Tensor, limit: float) -> torch.Tensor:
    """Optimal assignment for every batch element.  cost [B,R,C] float32, row_mask [B,R] bool,
    col_mask [B,C] bool -> [B,R] int64 column per row (-1 = unmatched / masked)."""
    if cost.is_cuda:
        return _load_ext().lap_batch_cuda(cost.float(), row_mask, col_mask, float(limit))
    return lap_batch_torch(cost, row_mask, col_mask, limit)


@lap_batch.register_fake
def _(cost, row_mask, col_mask, limit):
    return cost.new_empty(cost.shape[:2], dtype=torch.long)


# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def lap_batch_torch(cost: torch.Tensor, row_mask: torch.Tensor, col_mask: torch.Tensor,
                    limit: float) -> torch.Tensor:
    """Same algorithm, vectorised over the batch in plain PyTorch (any device).

    Rows are inserted in lock-step over the batch; the Dijkstra loop ends when every batch element
    has reached a free column, which needs a ``.any()`` host sync per step.
    """
    B, R, C = cost.shape
    dev = cost.device
    out = torch.full((B, R), -1, dtype=torch.long, device=dev)
    if B == 0 or R == 0 or C == 0:
        return out
    c = cost.double()
    allowed = row_mask[:, :, None] & col_mask[:, None, :] & (c <= limit)
    rok = allowed.any(2)                                   # [B,R]
    cok = allowed.any(1)                                   # [B,C]
    n = rok.sum(1)
    m = cok.sum(1)
    nmax, mmax = int(n.max()), int(m.max())
    if nmax == 0 or mmax == 0:
        return out
    # order-preserving compaction: stable sort puts selected indices first
    ri = torch.sort((~rok).to(torch.int8), dim=1, stable=True).indices[:, :nmax]   # [B,nmax]
    cj = torch.sort((~cok).to(torch.int8), dim=1, stable=True).indices[:, :mmax]
    sub = c.gather(1, ri[:, :, None].expand(B, nmax, C)).gather(2, cj[:, None, :].expand(B, nmax, mmax))
    ok = allowed.gather(1, ri[:, :, None].expand(B, nmax, C)).gather(2, cj[:, None, :].expand(B, nmax, mmax))
    ar_n = torch.arange(nmax, device=dev)
    ar_m = torch.arange(mmax, device=dev)
    ok &= (ar_n[None, :, None] < n[:, None, None]) & (ar_m[None, None, :] < m[:, None, None])
    INF = float("inf")
    # columns: 0 virtual, 1..mmax real, mmax+1..mmax+nmax dummy (dummy k valid iff k < n[b])
    W = mmax + nmax
    a = torch.full((B, nmax + 1, W + 1), INF, dtype=torch.float64, device=dev)
    a[:, 1:, 1:mmax + 1] = torch.where(ok, sub - limit, INF)
    dvalid = ar_n[None, :] < n[:, None]                                         # [B,nmax]
    a[:, 1:, mmax + 1:] = torch.where(dvalid[:, None, :], 0.0, INF)
    colvalid = torch.ones(B, W + 1, dtype=torch.bool, device=dev)
    colvalid[:, 1:mmax + 1] = ar_m[None, :] < m[:, None]
    colvalid[:, mmax + 1:] = dvalid
    colvalid[:, 0] = False
    u = torch.zeros(B, nmax + 1, dtype=torch.float64, device=dev)
    v = torch.zeros(B, W + 1, dtype=torch.float64, device=dev)
    p = torch.zeros(B, W + 1, dtype=torch.long, device=dev)
    way = torch.zeros(B, W + 1, dtype=torch.long, device=dev)
    bidx = torch.arange(B, device=dev)
    colidx = torch.arange(W + 1, device=dev)
    for i in range(1, nmax + 1):
        act = i <= n                                                            # [B]
        p[:, 0] = torch.where(act, i, 0)
        minv = torch.full((B, W + 1), INF, dtype=torch.float64, device=dev)
        used = torch.zeros(B, W + 1, dtype=torch.bool, device=dev)
        j0 = torch.zeros(B, dtype=torch.long, device=dev)
        running = act.clone()
        while bool(running.any()):
            used[bidx, j0] |= running
            i0 = p[bidx, j0]                                                    # [B]
            cur = a[bidx, i0] - u[bidx, i0][:, None] - v                        # [B,W+1]
            upd = running[:, None] & ~used & colvalid & (cur < minv)
            minv = torch.where(upd, cur, minv)
            way = torch.where(upd, j0[:, None], way)
            cand = torch.where(~used & colvalid, minv, INF)
            delta, j1 = cand.min(1)                                             # first index on ties
            delta = torch.where(running, delta, 0.0)
            urow = torch.where(used & running[:, None], p, 0)
            u.scatter_add_(1, urow, torch.where(used & running[:, None], delta[:, None], 0.0)
                           .expand_as(urow).contiguous())
            v = torch.where(used & running[:, None], v - delta[:, None], v)
            minv = torch.where(~used & running[:, None], minv - delta[:, None], minv)
            j0 = torch.where(running, j1, j0)
            running = running & (p[bidx, j0] != 0)
        # augment along way[] (sequential pointer chase, vectorised over the batch)
        cur_j = torch.where(act, j0, 0)
        while bool((cur_j != 0).any()):
            nz = cur_j != 0
            jp = way[bidx, cur_j]
            newp = p[bidx, jp]
            p[bidx, cur_j] = torch.where(nz, newp, p[bidx, cur_j])
            cur_j = torch.where(nz, jp, cur_j)
    # rows assigned to real columns
    pr = p[:, 1:mmax + 1]                                                       # [B,mmax] row (1-based)
    hit = (pr > 0) & (colidx[None, 1:mmax + 1] - 1 < m[:, None])
    rows = ri.gather(1, (pr - 1).clamp(min=0))
    tmp = torch.full((B, R + 1), -1, dtype=torch.long, device=dev)
    tmp.scatter_(1, torch.where(hit, rows, R), torch.where(hit, cj, -1))
    out.copy_(tmp[:, :R])
    return out


# ----------------------------------------------------------------------------------------------
def extended_cost(cost: np.ndarray, rows: np.ndarray, cols: np.ndarray, limit: float) -> float:
    """Objective of lap's extended problem for a (partial) assignment of the active submatrix."""
    n_r, n_c = cost.shape
    matched = len(rows)
    return float(cost[rows, cols].astype(np.float64).sum()) + limit / 2 * (n_r - matched + n_c - matched)
