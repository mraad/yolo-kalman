# yolo-kalman — native YOLO + GPU Kalman multi-object tracker with occlusion-robust ids

YOLO (Ultralytics YOLO26m) is used **only as the detector**. The tracker is my own code
(no `model.track()`, no boxmot). It runs in PyTorch on the GPU, batched over tracks and over
streams:

* **Kalman filter**: constant velocity on (cx, cy, w, h). BoT-SORT-style noise, NSA measurement noise
  `R·(1−conf)`, exact multi-step prediction (`kalman.py`). Tested against a NumPy reference.
* **Association cascade**: BYTE two-stage (high, then low score) with IoU + Mahalanobis (χ² 0.99) gating.
  OC-SORT **OCM** (direction-consistency term), **OCR** (last-observation IoU) and **ORU**
  (re-update along a virtual linear trajectory over the gap when a lost track is matched again).
  Appearance stage (S4) for long occlusions, and *young-track id recovery*: a freshly born track
  whose embedding matches a track that was already lost inherits the old id.
* **Lost-track buffer**: tracks coast on the KF prediction for `lost_buffer_s` (default 8 s), with size
  velocity frozen while lost.
* **Partial-occlusion handling** (needed to make the full-occlusion cases work with a real detector):
  a box much smaller than the track's slow-EMA reference size is a *low-quality* observation. The
  track coasts through it (R×1e4, no size velocity), and it is not used as an OC-SORT observation.
* **Appearance re-ID**: torchvision ResNet-18 (ImageNet) on RoI-aligned 128×64 crops taken straight from
  the decoded CUDA frames, giving 512-d features with a dynamic EMA per track. **Choice:** OSNet/torchreid
  weights are person-only (we also track vehicles/animals), distributed via Google Drive, and torchreid
  is unmaintained on PyPI. A generic class-agnostic descriptor installs from the PyTorch index. It sits
  behind a one-method interface (`reid.AppearanceEncoder`), so a proper re-ID net can replace it.
* **Assignment**: all cost matrices are computed on the GPU. The rectangular assignment uses `lap.lapjv`
  on the CPU (measured below: this CPU part is now the throughput limiter at ≥8 streams/GPU).

## GPU pipeline (`pipeline.py`)
```
NVDEC (PyNvVideoCodec ThreadedDecoder, frames stay in device memory, DLPack -> torch)
 -> GPU letterbox + normalise into one [S*T,3,384,640] fp16 batch          (S streams x T=8 frames)
 -> TensorRT FP16 YOLO26m, NMS-free end2end head [B,150,6] (no host sync)  (one batched call)
 -> un-letterbox, RoI-align crops -> ResNet-18 fp16 embeddings (one call for all streams)
 -> tracker step per frame for all S streams at once:
      CUDA graph A (predict + all cost terms -> pinned host) -> CPU cascade (lapjv) ->
      CUDA graph B (masked KF update / ORU / init / EMA -> pinned host)
```
The producer (decode/detect/embed) runs in its own thread on a low-priority CUDA stream. The tracker
runs on a high-priority stream. The tracker's dense GPU code is `torch.compile`d (inductor) and then
captured in CUDA graphs, with an element-wise 4×4 Cholesky so inductor can fuse it. On 16 streams this
cut tracker GPU time per step from 1.65 ms to 0.5 ms. Images never leave the GPU (only the optional
overlay writer downloads frames).

**What works for decode:** PyNvVideoCodec 2.2.4 NVDEC decodes the VP9 WebM clips and the MPEG-4
synthetic clips: about 5,000 fps for one 480p stream and about 20,000 fps aggregate for 16 decoders.
torchcodec was not needed (it also needs system FFmpeg libs, which are not installed). OpenCV CPU decode
is the fallback.

**Detector:** `scripts/export_trt.py` exports to TensorRT FP16 with a dynamic batch of up to 64 at
384×640. Both the NMS-free end2end head (default) and the raw one-to-many head (with our own one-call
GPU `batched_nms`) work, and outputs match Ultralytics `predict`. The TensorRT export **worked**; the
torch FP16 fallback (`TorchYolo`) exists but was not needed. The engine alone does about 3,300 img/s
per GPU (batch 32–64). The detector sits behind the small `Detector` protocol (`detector.py`):
letterboxed fp16 batch in, padded GPU tensors out. A permissive detector (YOLOX, RT-DETR/D-FINE) can be
dropped in. I did not build one; see "not done".

## Commands
```bash
~/.local/bin/uv sync                                   # env (torch 2.14+cu130, TensorRT 11.5, PyNvVideoCodec)
./run.sh scripts/download_clips.py                     # Wikimedia 480p VP9 transcodes -> clips/ (untrusted data)
cd models && ../run.sh ../scripts/export_trt.py yolo26m.pt --nms-free && cd ..   # TRT FP16 engine
./run.sh scripts/make_synthetic.py                     # synthetic occlusion clips + GT -> data/synth
./run.sh -m pytest -q                                  # 14 tests (KF vs NumPy, synthetic scenes, e2e)
./run.sh scripts/eval_synth.py --overlay               # HOTA/IDF1/IDSW + ablations -> results/synth
./run.sh scripts/track.py clips/*.webm --out results/real --overlay   # all clips as parallel streams
./run.sh scripts/bench.py --streams 1 4 8 16 --gpus 0 1 --frames 1200 # throughput + NVML sampling
./run.sh scripts/bench_table.py "twogpu_s*"            # markdown table from bench JSONs
```
`run.sh` = `.venv/bin/python -I` with `LD_LIBRARY_PATH` unset. The host path puts cuDNN 9.20 ahead of
the venv's 9.24, which breaks every conv.

## Results

### Synthetic clips with ground truth (real detector, full pipeline)
People walk behind a grey bar that hides them completely for a known number of frames (bar widths
set for 10/30/60/120-frame gaps; a third person walks the other way and is hidden longer). A separate
clip has a head-on crossing plus an X crossing. Ground truth = objects with visibility ≥ 0.3. HOTA is
my re-implementation of the TrackEval algorithm (sanity-checked on toy cases); IDF1/MOTA/IDSW come from
motmetrics. The last column is my direct check: is the id 1 s after reappearing equal to the id before
the full occlusion?

| variant | clip | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | frag | GT ids | pred ids | same id after full occlusion (1 s later / immediately) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full (BYTE+OC-SORT+re-ID) | synth_gap010 | 0.873 | 0.879 | 0.868 | 0.949 | 0.900 | 0 | 5 | 3 | 3 | 3/3 / 3/3 |
| full (BYTE+OC-SORT+re-ID) | synth_gap030 | 0.863 | 0.869 | 0.857 | 0.944 | 0.889 | 0 | 5 | 3 | 3 | 3/3 / 3/3 |
| full (BYTE+OC-SORT+re-ID) | synth_gap060 | 0.854 | 0.860 | 0.849 | 0.944 | 0.891 | 0 | 6 | 3 | 3 | 3/3 / 3/3 |
| full (BYTE+OC-SORT+re-ID) | synth_gap120 | 0.875 | 0.876 | 0.875 | 0.965 | 0.931 | 0 | 4 | 3 | 3 | 3/3 / 3/3 |
| full (BYTE+OC-SORT+re-ID) | synth_crossing | 0.883 | 0.900 | 0.866 | 0.946 | 0.896 | 0 | 3 | 4 | 4 | 1/1 / 1/1 |
| no re-ID | synth_gap010 | 0.873 | 0.879 | 0.868 | 0.949 | 0.900 | 0 | 5 | 3 | 3 | 3/3 / 3/3 |
| no re-ID | synth_gap030 | 0.799 | 0.745 | 0.857 | 0.845 | 0.885 | 1 | 5 | 3 | 4 | 2/3 / 2/3 |
| no re-ID | synth_gap060 | 0.655 | 0.485 | 0.884 | 0.618 | 0.931 | 5 | 4 | 3 | 6 | 0/3 / 0/3 |
| no re-ID | synth_gap120 | 0.637 | 0.462 | 0.878 | 0.568 | 0.928 | 3 | 4 | 3 | 6 | 0/3 / 0/3 |
| no re-ID | synth_crossing | 0.864 | 0.849 | 0.879 | 0.880 | 0.902 | 1 | 3 | 4 | 5 | 1/1 / 1/1 |
| no ORU/OCM/OCR | synth_gap010 | 0.839 | 0.775 | 0.908 | 0.829 | 0.949 | 1 | 6 | 3 | 4 | 3/3 / 3/3 |
| no ORU/OCM/OCR | synth_gap030 | 0.832 | 0.770 | 0.900 | 0.830 | 0.943 | 1 | 7 | 3 | 4 | 3/3 / 3/3 |
| no ORU/OCM/OCR | synth_gap060 | 0.840 | 0.848 | 0.833 | 0.938 | 0.879 | 0 | 6 | 3 | 3 | 3/3 / 3/3 |
| no ORU/OCM/OCR | synth_gap120 | 0.876 | 0.876 | 0.875 | 0.965 | 0.931 | 0 | 4 | 3 | 3 | 3/3 / 3/3 |
| no ORU/OCM/OCR | synth_crossing | 0.919 | 0.923 | 0.916 | 0.970 | 0.944 | 2 | 2 | 4 | 5 | 1/1 / 1/1 |
| ByteTrack-like (1 s buffer, no OC-SORT, no re-ID) | synth_gap010 | 0.831 | 0.761 | 0.906 | 0.860 | 0.942 | 3 | 6 | 3 | 5 | 2/3 / 2/3 |
| ByteTrack-like (1 s buffer, no OC-SORT, no re-ID) | synth_gap030 | 0.679 | 0.512 | 0.899 | 0.653 | 0.936 | 5 | 6 | 3 | 7 | 0/3 / 0/3 |
| ByteTrack-like (1 s buffer, no OC-SORT, no re-ID) | synth_gap060 | 0.653 | 0.483 | 0.884 | 0.618 | 0.927 | 5 | 6 | 3 | 7 | 0/3 / 0/3 |
| ByteTrack-like (1 s buffer, no OC-SORT, no re-ID) | synth_gap120 | 0.637 | 0.462 | 0.878 | 0.568 | 0.928 | 3 | 4 | 3 | 6 | 0/3 / 0/3 |
| ByteTrack-like (1 s buffer, no OC-SORT, no re-ID) | synth_crossing | 0.925 | 0.926 | 0.923 | 0.968 | 0.943 | 4 | 2 | 4 | 5 | 1/1 / 1/1 |

The full tracker kept **13/13 full occlusions with the same id (gaps up to 157 frames), with 0 ID
switches on every clip**. Without re-ID, motion alone carries the id through about 10–30 frames of
full occlusion. Gaps of 60 frames or more fail, because people do not reappear exactly where a
constant-velocity model puts them when their box was partially eroded before vanishing. The
ByteTrack-like baseline fails from 30 frames on.

Unit tests with simulated detections (`tests/test_synthetic.py`) cover gaps of 10/30/60/120 frames
(4 scenes as 4 batched streams, with and without re-ID), a speed change behind the occluder (needs
re-ID), crossings with no swap, clutter (HOTA > 0.7, IDF1 > 0.85, IDSW = 0), and CUDA-graph/compiled
output identical to eager.

### Real clips (no ground truth)
12 open-licence Wikimedia Commons clips (480p VP9 transcodes), tracked as 12 parallel streams on one
GPU, with classes person, bicycle, car, motorcycle, bus, truck, bird, cat and dog. *Fragmentations* =
number of times a track id disappears and later reappears. With re-ID many more objects are *re-acquired
under their old id*, so fragmentations go **up** while the number of ids goes **down**. Without ground
truth I cannot rule out that some of the extra re-acquisitions are wrong merges; that is unverified.
Overlays with ids and trails: `results/real/*_tracks.mp4` and `results/demo/` (not in git).

| clip | licence | frames | ids (full) | ids (no re-ID) | ids ≥1 s (full / no re-ID) | fragmentations (full / no re-ID) | re-appear after ≥1 s gap (full / no re-ID) | max gap bridged (frames) |
|---|---|---|---|---|---|---|---|---|
| Calle_preciados.webm | CC BY-SA 3.0 | 456 | 37 | 39 | 20 / 36 | 145 / 69 | 24 / 22 | 138 |
| Inbound_B_Branch_train__March_2021.webm | CC BY 2.0 | 2100 | 107 | 139 | 50 / 66 | 441 / 251 | 81 / 49 | 215 |
| Lisboa_Carris__40267618395.webm | CC BY 2.0 | 225 | 19 | 20 | 11 / 12 | 35 / 22 | 2 / 2 | 104 |
| London_United_LTZ1130_passing_House_of_Commons_i | CC BY 3.0 | 116 | 23 | 30 | 6 / 5 | 37 / 8 | 5 / 1 | 87 |
| MD_355_and_MD_124_Gaithersburg_MD_2022_07_30_11_ | CC BY 4.0 | 1974 | 85 | 126 | 42 / 103 | 734 / 363 | 137 / 58 | 238 |
| Roncesvalles_Ave_crossing_at_Galley_Ave__PXO_Typ | CC BY-SA 4.0 | 390 | 13 | 14 | 11 / 12 | 18 / 17 | 2 / 2 | 61 |
| Soundscapes_of_Norrk_ping__1____Skvallertorget.w | CC BY 3.0 | 1702 | 84 | 107 | 43 / 55 | 338 / 170 | 53 / 18 | 237 |
| Soundscapes_of_Norrk_ping__4____Train_station.we | CC BY 3.0 | 1994 | 59 | 87 | 32 / 50 | 291 / 145 | 47 / 18 | 235 |
| Starling_bathing_one_sparrow_eating_two_doves_wa | CC BY 3.0 | 10540 | 27 | 39 | 14 / 18 | 224 / 148 | 43 / 22 | 434 |
| Strassenbahn_Gmunden_in_der_Alois_Kaltenbruner_S | CC BY 3.0 | 3521 | 53 | 84 | 18 / 21 | 257 / 166 | 28 / 11 | 214 |
| Traffic_on_Slovenska_street.webm | CC BY 3.0 | 1353 | 74 | 116 | 43 / 75 | 335 / 117 | 76 / 23 | 194 |
| Traffic_on_Slovenska_street_2.webm | CC BY 3.0 | 466 | 22 | 32 | 15 / 17 | 79 / 55 | 12 / 11 | 77 |
| **total** | | 24837 | **603** | 833 | | | 510 / 237 | |

Tracker counters for the full run: 3,291 ORU re-updates, 1,217 appearance re-IDs (S4), 729 OCR
matches, 175 young-track id recoveries, 0 dropped detections, 0 slot overflows.

### Throughput (end-to-end: NVDEC → TRT → re-ID → tracker, no overlay)
`scripts/bench.py`: the 5 longest clips repeated round-robin, 1,200 frames per stream, one worker
process per GPU (streams split across GPU 0/1). NVML is sampled every 100 ms during the run. The time
split is the per-worker sum of stage wall time. Producer (decode/letterbox/detect/embed) and consumer
(track) overlap, so the columns add up to more than the wall time.

| streams | GPUs | frames | wall s | FPS total | FPS/stream | GPU util % (mean per GPU) | NVDEC % | decode s | letterbox s | detect s | embed s | track s | tracker GPU-pre / CPU-match / GPU-post s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1200 | 1.4 | 845 | 845 | 40 | 3 | 0.05 | 0.03 | 0.59 | 0.64 | 1.30 | 0.36 / 0.65 / 0.20 |
| 4 | 2 | 4800 | 2.1 | 2268 | 567 | 47 / 45 | 6 / 6 | 0.09 | 0.06 | 0.88 | 0.69 | 1.81 | 0.50 / 0.88 / 0.31 |
| 8 | 2 | 9600 | 3.6 | 2669 | 334 | 51 / 48 | 7 / 7 | 0.18 | 0.09 | 1.59 | 0.76 | 3.36 | 0.59 / 1.86 / 0.72 |
| 16 | 2 | 19200 | 6.4 | 3015 | 188 | 55 / 54 | 7 / 7 | 0.35 | 0.27 | 3.05 | 0.85 | 6.06 | 0.86 / 3.35 / 1.65 |
| 32 | 2 | 38400 | 10.9 | 3519 | 110 | 60 / 61 | 8 / 8 | 0.69 | 0.57 | 6.05 | 0.89 | 10.65 | 1.31 / 6.61 / 2.53 |

Two worker processes per GPU (`--wpg 2`):

| streams | GPUs | frames | wall s | FPS total | FPS/stream | GPU util % (mean per GPU) | NVDEC % | decode s | letterbox s | detect s | embed s | track s | tracker GPU-pre / CPU-match / GPU-post s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 2 | 19200 | 5.7 | 3378 | 211 | 93 / 91 | 10 / 10 | 0.56 | 0.57 | 2.89 | 0.74 | 5.09 | 1.82 / 1.81 / 1.23 |
| 32 | 2 | 38400 | 11.2 | 3432 | 107 | 94 / 92 | 10 / 10 | 1.27 | 1.51 | 7.37 | 0.66 | 9.33 | 3.55 / 3.47 / 1.99 |


**Bottleneck analysis (iterations, in order):**
1. Eager tracker with per-track ORU loop: 158 fps for 1 stream, 533 fps for 8 streams on 1 GPU. GPU was
   20–32% busy. Bottleneck: Python kernel launches (≈1,000 tiny kernels per step).
2. Dense fixed-shape slot tensors + 2 CUDA graphs per step: 686 fps for 1 stream, 1,313 fps for 16
   streams on 1 GPU.
3. Batched embeddings across streams (embed time 3.2 s → 0.4 s) and NMS-free YOLO26 head (no host
   sync): 1,554 fps for 16 streams on 1 GPU.
4. `torch.compile` + unrolled 4×4 Cholesky inside the graphs, and a high-priority tracker stream:
   1,943 fps for 16 streams on 1 GPU; **3,015 fps (16 streams) / 3,519 fps (32 streams) on 2 GPUs**.
5. Now `track ≈ wall`: the consumer thread is the limiter. Of that time, the CPU matching cascade
   (lapjv + NumPy bookkeeping, ~0.2 ms per stream per frame) is the largest part. The rest is waiting
   on the two graph syncs while the GPU is busy with TensorRT. The detector alone could do about 3,300
   img/s per GPU, so the end-to-end ceiling would be about 6,600 fps on 2 GPUs; we reach 53%.
   NVML "utilisation" is 55–60% with one worker per GPU. Two workers per GPU push NVML util to 93%
   but gain only 12% at 16 streams and nothing at 32: without MPS the processes time-slice the GPU,
   and every tracker step has two blocking syncs. Next steps that would help (not done): a GPU
   auction/greedy assignment so the cascade stays on the GPU, pipelining two stream groups so one
   group's CPU matching overlaps the other's graphs, or MPS.

## Licences
* **Ultralytics YOLO26 / `ultralytics` package: AGPL-3.0** (accepted for this project). Anything that
  distributes or serves this tracker with the Ultralytics detector inherits AGPL obligations. The
  detector is isolated behind `detector.Detector` so a permissively licensed one (YOLOX Apache-2.0,
  RT-DETR/D-FINE Apache-2.0 via HF transformers) can replace it. The synthetic clips use person sprites
  cut from Ultralytics' bundled `bus.jpg`/`zidane.jpg` (AGPL-3.0 package assets).
* Own tracker code: no licence chosen (the user's call). torchvision ResNet-18 weights: BSD-3 (torchvision).
* Clips: see `meta/clips_index.tsv` (CC BY 2.0/3.0/4.0, CC BY-SA 3.0/4.0, with Wikimedia Commons
  description pages for attribution). They are not committed. Two listed clips (Hofer Verkehrswendetag
  C0258/C0270) have no Wikimedia transcodes (404 for every variant), and their 4K originals are
  2.6 GB / 2.0 GB, so I skipped them.

## Not done / known limitations
* No permissive replacement detector was built (time went into GPU throughput).
* Real-clip quality is unverified without ground truth (id counts and fragmentations only).
* ResNet-18 ImageNet features are not a trained re-ID model. They work on the synthetic clips and
  noticeably reduce id counts on real clips, but similar-looking objects (same-colour cars) can be
  confused. The S4/recovery stages are gated by Mahalanobis distance and class group to limit this.
* HOTA is a re-implementation, not TrackEval itself.
* Ids are unique per stream, not globally.
