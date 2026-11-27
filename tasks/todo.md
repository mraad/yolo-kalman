# yolo-kalman todo

## Setup
- [x] git init, .gitignore (exclude clips/, results/ videos, models)
- [x] uv project: torch 2.14+cu130, ultralytics 8.4, tensorrt-cu13 11.5, PyNvVideoCodec 2.2.4
- [x] run.sh wrapper (clean LD_LIBRARY_PATH: host cuDNN 9.20 broke torch convs; `python -I`)
- [x] CLAUDE.md

## Data
- [x] download Wikimedia 480p VP9 transcodes (generic UA) -> 12/14 clips (2 Hofer clips: no transcode, 4K originals 2.0-2.6 GB, skipped)
- [x] synthetic occlusion clips with GT (grey bar, gaps 10/30/60/120 + crossing) -> data/synth

## Detector
- [x] Detector interface (letterboxed GPU batch -> padded GPU tensors)
- [x] YOLO26m TensorRT FP16 export, dynamic batch <= 64, 384x640 (raw head + GPU NMS, and NMS-free end2end head)
- [x] torch FP16 fallback backend (TorchYolo)
- [x] GPU letterbox/normalise

## Decode
- [x] NVDEC via PyNvVideoCodec ThreadedDecoder (VP9 + MPEG-4 ok); OpenCV fallback

## Tracker (GPU, torch)
- [x] batched CV Kalman filter (cx,cy,w,h), NSA noise, exact multi-step predict
- [x] BYTE two-stage association, IoU + Mahalanobis gating
- [x] lost-track buffer (seconds), OC-SORT ORU + OCM (+OCR)
- [x] appearance re-ID (ResNet-18 RoI features, EMA), long-occlusion S4 re-ID + young-track id recovery
- [x] batched over streams; CUDA graphs + torch.compile for the dense GPU part
- [x] partial-occlusion handling (low-quality obs coast; size reference EMA)

## Tests
- [x] KF vs NumPy reference (+ multi-step predict, unrolled 4x4 Cholesky)
- [x] synthetic scenes: gaps 10/30/60/120 (batched streams), crossings, speed change, clutter
- [x] CUDA-graph/compiled path == eager path
- [x] e2e on short real clip (+ rendered synthetic clip with real detector)

## Metrics / bench
- [x] HOTA/IDF1/IDSW on synthetic GT clips + ablations (results/synth/metrics.md)
- [x] real clips: id count, fragmentation, overlays (results/real_table.md)
- [x] benchmark 1/4/8/16/32 streams over 2 GPUs, NVML util, time split
- [x] iterate on bottleneck / document it (CPU matching cascade in tracker consumer)

## Wrap-up
- [x] README (commands, results, licences)
- [x] demo in tmux `tracker-demo` -> results/demo (12 overlays, exit 0, session closed)
- [x] DONE summary

## GPU Hungarian
Decisions (made unattended):
- Method: exact Hungarian / Kuhn-Munkres, shortest-augmenting-path form with dual potentials (Dijkstra
  per row, e-maxx/JV family), as a custom CUDA kernel: one warp per stream problem, all streams in one
  launch, data-dependent loops stay inside the kernel -> CUDA-graph capturable, sync-free.
  A pure-PyTorch batched variant needs a host sync per Dijkstra step (data-dependent termination) ->
  kept only as a reference/fallback and measured.
- cost_limit semantics = lap's extend_cost: minimise sum(c_ij - limit) over matched pairs, so pairs
  with c > limit are forbidden; implemented with n zero-cost dummy columns (row unmatched).
- Whole cascade (S1..S5, new tracks, id recovery, bookkeeping) ported to GPU tensors for
  assign="gpu_hungarian"; assign="lapjv" keeps the old CPU cascade byte-for-byte as the reference.
- [x] CUDA kernel + torch custom op + torch fallback (torch variant measured 300x slower: host sync per Dijkstra step)
- [x] solver tests vs scipy / lapjv (tests/test_assign.py, 7 tests)
- [x] GPU cascade in tracker, assign option, equivalence vs lapjv path (frame-identical on synthetic scenes)
- [x] async step (one sync per chunk) in pipeline
- [x] CLI flags track.py / bench.py / eval_synth.py
- [x] bench both solvers 1/4/8/16/32 streams, 2 GPUs (5,270 vs 3,164 fps at 16)
- [x] README / CLAUDE.md / DONE_HUNGARIAN
- [x] iterate: fusable KF (no 8x8 bmm), solver kernel shared-mem staging, split producer (reader + detector)
- [x] found + fixed: NVDEC copy-out race (exposed by sync-free reader), benchmark lost-"done" hang
- Decision: default assign="gpu_hungarian" (exact, output-identical to lapjv on synthetic suites, 1.4-1.7x faster end to end).
- [x] real clips: recorded-input replay, every solver call vs lapjv: equal total cost 8,536/8,536, 222 tie
  alternatives; full runs 591/595 ids (gpu) vs 619/601 (lapjv): run-to-run noise (pipeline not deterministic)
- Not done: fused single-kernel cascade (next step if more tracker speed is needed under TRT contention);
  root cause of 3 fast rc=1 benchmark failures right after the queue-flush fix not recorded.
