# yolo-kalman todo

## Setup
- [ ] git init, .gitignore (exclude clips/, results/ large files, agent harness files)
- [ ] uv project, torch (Blackwell / cu13x), ultralytics, tensorrt, torchcodec/PyNvVideoCodec
- [ ] tasks/todo.md, CLAUDE.md

## Data
- [ ] download Wikimedia 480p transcodes (generic UA) into clips/, licence table
- [ ] synthetic occlusion clip with GT (grey bar)

## Detector
- [ ] Detector interface (letterboxed GPU batch -> boxes/scores/classes on GPU)
- [ ] YOLO11/26 TensorRT FP16 export (fallback torch FP16)
- [ ] GPU letterbox/normalise

## Decode
- [ ] NVDEC decode (torchcodec CUDA / PyNvVideoCodec), record what works

## Tracker (GPU, torch)
- [ ] batched CV Kalman filter (cx,cy,w,h), NSA noise
- [ ] BYTE two-stage association, IoU + Mahalanobis gating
- [ ] lost-track buffer, OC-SORT ORU + OCM
- [ ] appearance re-ID EMA embedding, long-occlusion re-acquisition
- [ ] batched over streams

## Tests
- [ ] KF vs NumPy reference
- [ ] synthetic scenes: gaps 10/30/60/120, crossings
- [ ] e2e on short real clip

## Metrics / bench
- [ ] HOTA/IDF1/IDSW on synthetic GT clips
- [ ] real clips: id count, fragmentation, overlays
- [ ] benchmark 1/4/8/16 streams over 2 GPUs, NVML util, time split
- [ ] iterate on bottleneck

## Wrap-up
- [ ] README (commands, results, licences)
- [ ] demo in tmux `tracker-demo` -> results/demo, stop when done
- [ ] DONE summary
