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

CUDA kernel: one thread block per batch element (= stream).  All 256 threads find the allowed
rows/columns (coalesced pass over the padded matrix) and stage the compacted sub-matrix in shared
memory when it fits the device's per-block shared-memory limit (larger problems read global
memory); then one warp runs the Hungarian search: columns strided over the 32 lanes, the Dijkstra
arg-min is a warp-shuffle reduction (ties -> lowest column index, deterministic).  All
data-dependent loops live inside the kernel, so a call is one fixed launch: CUDA-graph capturable,
no host sync.  Exposed as the torch custom op ``yolo_kalman::lap_batch`` (works under
torch.compile).  A pure-PyTorch batched version was measured ~300x slower (host sync per Dijkstra
step) and removed (see README).
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
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>

#define FULL 0xffffffffu

#define NT 256          // threads per block: all of them compact / stage, warp 0 solves

// one block per batch element
__global__ void __launch_bounds__(NT) lap_sap_kernel(
    const float* __restrict__ cost, const bool* __restrict__ rmask, const bool* __restrict__ cmask,
    int64_t* __restrict__ out, int R, int C, double limit, int submax) {
  extern __shared__ unsigned char smem[];
  const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
  const float* cb = cost + (size_t)b * R * C;
  const bool* rm = rmask + (size_t)b * R;
  const bool* cm = cmask + (size_t)b * C;
  int64_t* ob = out + (size_t)b * R;
  const int WMAX = R + C;
  // shared layout (8-byte arrays first)
  double* u    = (double*)smem;                 // [R+1]   row duals (1-based)
  double* v    = u + (R + 1);                   // [W+1]   column duals (1-based, 0 = virtual)
  double* minv = v + (WMAX + 1);                // [W+1]
  float* sub   = (float*)(minv + (WMAX + 1));   // [submax] compacted costs (row-major n x m)
  int* p       = (int*)(sub + submax);          // [W+1]   row assigned to column (0 = free)
  int* way     = p + (WMAX + 1);                // [W+1]   predecessor column on the path
  int* ri      = way + (WMAX + 1);              // [R]     compacted row -> original row
  int* cj      = ri + R;                        // [C]     compacted col -> original col
  int* nm      = cj + C;                        // [2]     n, m
  unsigned char* used = (unsigned char*)(nm + 2);  // [W+1]
  unsigned char* rowf = used + (WMAX + 1);      // [R]
  unsigned char* colf = rowf + R;               // [C]

  for (int r = tid; r < R; r += NT) { ob[r] = -1; rowf[r] = 0; }
  for (int k = tid; k < C; k += NT) colf[k] = 0;
  __syncthreads();
  // ---- allowed pairs (coalesced over the whole block): row/col has >= 1 allowed pair ----
  for (int e = tid; e < R * C; e += NT) {
    int r = e / C, k = e - r * C;
    if (rm[r] && cm[k] && (double)cb[e] <= limit) { rowf[r] = 1; colf[k] = 1; }
  }
  __syncthreads();
  // ---- order-preserving compaction (warp 0, ballots over the flags) ----
  if (tid < 32) {
    const unsigned lt = (1u << lane) - 1u;
    int n = 0;
    for (int base = 0; base < R; base += 32) {
      int r = base + lane;
      bool f = r < R && rowf[r];
      unsigned bal = __ballot_sync(FULL, f);
      if (f) ri[n + __popc(bal & lt)] = r;
      n += __popc(bal);
    }
    int m = 0;
    for (int base = 0; base < C; base += 32) {
      int k = base + lane;
      bool f = k < C && colf[k];
      unsigned bal = __ballot_sync(FULL, f);
      if (f) cj[m + __popc(bal & lt)] = k;
      m += __popc(bal);
    }
    if (lane == 0) { nm[0] = n; nm[1] = m; }
  }
  __syncthreads();
  const int n = nm[0], m = nm[1];
  if (n == 0 || m == 0) return;
  // stage the compacted costs in shared memory when they fit (submax, set by the host from the
  // device limit); larger problems read the padded matrix in global memory through the column index
  const bool staged = n * m <= submax;
  if (staged) {
    for (int e = tid; e < n * m; e += NT) {
      int i = e / m, j = e - i * m;
      sub[e] = cb[(size_t)ri[i] * C + cj[j]];
    }
  }
  __syncthreads();
  if (tid >= 32) return;      // the Hungarian search is warp-synchronous from here on
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
      const float* srow = sub + (size_t)(i0 - 1) * m;
      double delta = INF;
      int j1 = 0x7fffffff;
      for (int j = 1 + lane; j <= W; j += 32) {
        if (!used[j]) {
          double a;
          if (j <= m) {
            float c = staged ? srow[j - 1] : row[cj[j - 1]];
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
  const c10::cuda::CUDAGuard guard(c.device());
  // fixed shared arrays of the kernel layout (+16 B alignment slack), then the staged-cost budget
  size_t base = (size_t)(R + 1) * 8 + (size_t)(W + 1) * 16 + (size_t)(W + 1) * 8
                + (size_t)(R + C + 2) * 4 + (W + 1) + R + C + 16;
  const size_t maxsh = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;  // cached by ATen
  TORCH_CHECK(base <= maxsh, "lap_batch: ", R, "x", C, " needs ", base, " B shared memory, device allows ",
              maxsh, " B: lower max_tracks / max_dets");
  const int submax = (int)std::min<size_t>((size_t)R * C, (maxsh - base) / 4);
  size_t sh = base + (size_t)submax * 4;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (sh > 48 * 1024)
    cudaFuncSetAttribute(lap_sap_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
  lap_sap_kernel<<<B, NT, sh, stream>>>(c.data_ptr<float>(), rm.data_ptr<bool>(), cmk.data_ptr<bool>(),
                                        out.data_ptr<int64_t>(), R, C, limit, submax);
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
    if not cost.is_cuda:
        raise ValueError("lap_batch needs CUDA tensors (use lap.lapjv / scipy on the CPU)")
    return _load_ext().lap_batch_cuda(cost.float(), row_mask, col_mask, float(limit))


@lap_batch.register_fake
def _(cost, row_mask, col_mask, limit):
    return cost.new_empty(cost.shape[:2], dtype=torch.long)


# ----------------------------------------------------------------------------------------------
def extended_cost(cost: np.ndarray, rows: np.ndarray, cols: np.ndarray, limit: float) -> float:
    """Objective of lap's extended problem for a (partial) assignment of the active submatrix."""
    n_r, n_c = cost.shape
    matched = len(rows)
    return float(cost[rows, cols].astype(np.float64).sum()) + limit / 2 * (n_r - matched + n_c - matched)
