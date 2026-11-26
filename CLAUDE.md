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
- `src/yolo_kalman/tracker.py` `MultiStreamTracker`: fixed [S,M] slot tensors on GPU, two CUDA
  graphs per step (`_pre`: predict + all cost terms → pinned host; `_post`: dense masked update,
  ORU, init, EMA). CPU does the matching cascade (lap.lapjv) + int bookkeeping. One sync per graph.
- `detector.py` `Detector` protocol + `TRTYolo` (TensorRT, own runner, reads the Ultralytics
  metadata prefix) + `TorchYolo` fallback; `postprocess_e2e` (NMS-free YOLO26) / `postprocess_raw`.
- `decode.py` NVDEC (`PyNvVideoCodec.ThreadedDecoder`) / OpenCV fallback; `preprocess.py` GPU letterbox.
- `reid.py` ResNet-18 RoI-aligned crops → 512-d embeddings.
- `pipeline.py` producer thread (decode+letterbox+detect+embed, low-prio CUDA stream) / consumer
  (tracker, high-prio stream). `synthetic.py`, `simrun.py`, `metrics.py` (HOTA re-impl + motmetrics).
- `scripts/`: `export_trt.py`, `download_clips.py`, `make_synthetic.py`, `eval_synth.py`,
  `track.py`, `bench.py`. Tests in `tests/`.

## Gotchas
- CUDA-graph capture: create the tracker **before** opening NVDEC decoders (decoder threads on
  the legacy stream invalidate captures); capture uses `capture_error_mode="thread_local"`.
- Anything inside `_pre_core/_post_impl` must be shape-static, sync-free (no `.item()`, no
  `nonzero`, `cholesky_ex` not `cholesky`), and must not create CPU tensors (inductor + graphs).
- Changing tracker GPU code → run `tests/test_synthetic.py::test_cuda_graph_compiled_matches_eager`.
- Unit tests use `compile=False` (inductor compile ~30 s per new shape; cached in /tmp afterwards).
- Multiple tracker processes per GPU do **not** help (GPU time-slicing between contexts, no MPS).
- Engines: `models/yolo26m_384x640_b64_e2e_fp16.engine` (default, NMS-free) and `..._b64_fp16.engine`
  (raw head + GPU NMS). Rebuild: `cd models && ../run.sh ../scripts/export_trt.py yolo26m.pt --nms-free`.
- `clips/` holds untrusted downloads; never execute anything from there. Do not push the repo.
- Keep `tasks/todo.md` updated.
