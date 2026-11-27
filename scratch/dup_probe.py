"""Count near-duplicate detections of the NMS-free head: pairs of valid boxes per frame by IoU bucket."""
import glob, sys, torch
from torchvision.ops import box_iou
from yolo_kalman.decode import NvDecSource
from yolo_kalman.detector import TRTYolo
from yolo_kalman.preprocess import letterbox_into

det = TRTYolo("models/yolo26m_384x640_b64_e2e_fp16.engine", conf=0.25)
edges = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.01], device="cuda")
same = torch.zeros(5, dtype=torch.long, device="cuda"); diff = same.clone(); frames = dets = 0
x = torch.empty(32, 3, 384, 640, dtype=torch.float16, device="cuda")
for path in sorted(glob.glob("clips/*.webm")):
    src = NvDecSource(path, chunk=32)
    for _ in range(40):
        fr = src.read(32)
        if fr is None or fr.shape[0] < 32: break
        letterbox_into(x, fr); d = det(x)
        for b in range(32):
            v = d.valid[b]; bx = d.boxes[b][v]; c = d.cls[b][v]
            iou = torch.triu(box_iou(bx, bx), 1); eq = c[:, None] == c[None, :]
            for k in range(5):
                m = (iou >= edges[k]) & (iou < edges[k + 1])
                same[k] += (m & eq).sum(); diff[k] += (m & ~eq).sum()
            frames += 1; dets += int(v.sum())
print("frames", frames, "dets", dets)
print("IoU bucket   0.5-0.6 0.6-0.7 0.7-0.8 0.8-0.9 0.9-1.0")
print("same class ", same.tolist()); print("other class", diff.tolist())
