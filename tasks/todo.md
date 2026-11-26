# yolo-kalman todo

## Setup
- [x] git init, .gitignore (exclude clips/, results/ videos, models)
- [x] uv project: torch 2.14+cu130, ultralytics 8.4, tensorrt-cu13 11.5, PyNvVideoCodec 2.2.4
- [x] run.sh wrapper (clean LD_LIBRARY_PATH: host cuDNN 9.20 broke torch convs; `python -I`)
- [ ] CLAUDE.md

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
- [ ] e2e on short real clip

## Metrics / bench
- [x] HOTA/IDF1/IDSW on synthetic GT clips + ablations (results/synth/metrics.md)
- [ ] real clips: id count, fragmentation, overlays
- [ ] benchmark 1/4/8/16 streams over 2 GPUs, NVML util, time split
- [ ] iterate on bottleneck / document it

## Wrap-up
- [ ] README (commands, results, licences)
- [ ] demo in tmux `tracker-demo` -> results/demo, stop when done
- [ ] DONE summary
