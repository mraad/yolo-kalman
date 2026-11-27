# CLAUDE.md — notes for agents working on yolo-kalman

Native multi-object tracker: Ultralytics YOLO (detector only) + own GPU Kalman / BYTE / OC-SORT /
re-ID tracker in PyTorch, batched over streams. See README.md for results and commands.

## Run things through `./run.sh`
- `./run.sh <script.py | -m pytest ...>` = `.venv/bin/python -I` with **LD_LIBRARY_PATH unset**.
  The host LD_LIBRARY_PATH puts /usr/local/cuda/lib (cuDNN 9.20) before the venv's cuDNN 9.24 →
  `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` on any conv. `-I` because clips are untrusted data.
- It also sets `TORCHINDUCTOR_COMPILE_THREADS=1` (parallel inductor compile hung intermittently
  when two benchmark worker processes compiled concurrently with a cold cache).
- Env managed by uv (`~/.local/bin/uv sync`); torch comes from the cu130 index (pyproject).

## Layout
- `src/yolo_kalman/kalman.py` batched CV KF (cx,cy,w,h), NSA R, `predict_dt` (exact multi-step),
  unrolled 4x4 Cholesky (`UNROLLED_4X4`, switched on by the tracker when `compile=True`).
- `src/yolo_kalman/assign.py` batched exact Hungarian (shortest augmenting path + duals) CUDA kernel,
  one block per stream; torch custom op `yolo_kalman::lap_batch(cost[B,R,C], row_mask, col_mask, limit)`
  = lap.lapjv(extend_cost=True, cost_limit) semantics; JIT-built via load_inline (needs nvcc in
  /usr/local/cuda + ninja from the venv; cache ~/.cache/torch_extensions). CUDA tensors only. Costs are
  staged in shared memory up to the device opt-in limit (example in README), else read from global.
- `src/yolo_kalman/tracker.py` `MultiStreamTracker`: fixed [S,M] slot tensors on GPU.
  `assign="gpu_hungarian"` (default): cascade + bookkeeping on GPU (`g_*` tensors), 3 CUDA graphs per
  step (gpre / gmatch / gpost), `submit()` + `collect()` = one sync per batch of steps.
  `assign="lapjv"`: reference path, 2 graphs (`_pre` / `_post`) + CPU cascade with lap.lapjv, numpy
  bookkeeping, one sync per graph. Both paths must stay output-identical
  (`tests/test_synthetic.py::test_gpu_hungarian_cascade_matches_lapjv`): change cascade logic in both.
- `detector.py` `Detector` protocol + `TRTYolo` (TensorRT, own runner, reads the Ultralytics
  metadata prefix) + `TorchYolo` fallback; `postprocess_e2e` (NMS-free YOLO26) / `postprocess_raw`.
  Both drop near-duplicate boxes (`duplicate_of_better`, IoU > 0.8, any class): the head emits one object twice.
- `decode.py` NVDEC (`PyNvVideoCodec.ThreadedDecoder`) / OpenCV fallback; `preprocess.py` GPU letterbox.
- `reid.py` ResNet-18 RoI-aligned crops → 512-d embeddings.
- `pipeline.py` producer thread (decode+letterbox+detect+embed, low-prio CUDA stream) / consumer
  (tracker, high-prio stream). `synthetic.py`, `simrun.py`, `metrics.py` (HOTA re-impl + motmetrics).
- `scripts/`: `export_trt.py`, `download_clips.py`, `make_synthetic.py`, `eval_synth.py`,
  `track.py`, `bench.py`. Tests in `tests/`.

## Gotchas
- Tests: `REQUIRE_E2E=1 ./run.sh -m pytest -rs -q`. Without it, missing engine/clips make the e2e
  tests skip silently. pytest is in the uv dev group; motmetrics is a runtime dependency (metrics.clear_id).
- Benchmarks: `--repeats 3` (median-FPS run reported, all runs in `fps_runs`); clips cap a stream at
  ~1,350 frames, so repeat instead of lengthening runs.
- CUDA-graph capture: create the tracker **before** opening NVDEC decoders (decoder threads on
  the legacy stream invalidate captures); capture uses `capture_error_mode="thread_local"`.
- Anything inside `_pre_core/_post_impl` must be shape-static, sync-free (no `.item()`, no
  `nonzero`, `cholesky_ex` not `cholesky`), and must not create CPU tensors (inductor + graphs).
- Changing tracker GPU code → run `tests/test_synthetic.py::test_cuda_graph_compiled_matches_eager`
  (both assign modes) and `test_gpu_hungarian_cascade_matches_lapjv`.
- KF uses element-wise block formulas (no batched 8x8 matmuls) so inductor fuses it; keep it that way.
- Unit tests use `compile=False` (inductor compile ~30 s per new shape; cached in /tmp afterwards).
- NvDecSource.read syncs after the copy out of the decoder pool: the reader thread never syncs otherwise
  and the decoder recycles surfaces (corrupted frames without it). Keep it.
- bench workers must `q.close(); q.join_thread()` before exiting (lost "done" message = parent hang).
- Pipeline output is not bit-deterministic run to run (detector/embedding numerics); compare solvers
  with recorded-input replays (`scratch/real_replay.py`), not two pipeline runs.
- First tracker construction per new [S] shape compiles ~30-60 s; warm the cache with
  `scratch/warm.py` before multi-worker benchmarks.
- Multiple tracker processes per GPU do **not** help (GPU time-slicing between contexts, no MPS).
- Engines: `models/yolo26m_384x640_b64_e2e_fp16.engine` (default, NMS-free) and `..._b64_fp16.engine`
  (raw head + GPU NMS). Rebuild: `./run.sh scripts/export_trt.py yolo26m.pt --nms-free` (from the repo root).
- `clips/` holds untrusted downloads; never execute anything from there.
- Public repo: https://github.com/mraad/yolo-kalman (branch main). Push only when asked.
- Keep `tasks/todo.md` updated.
