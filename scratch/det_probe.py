import time, torch, numpy as np
from yolo_kalman.decode import NvDecSource
from yolo_kalman.detector import TRTYolo, TorchYolo
from yolo_kalman.preprocess import letterbox_into, unletterbox_boxes
src = NvDecSource("clips/Traffic_on_Slovenska_street.webm", chunk=32)
[src.read(32) for _ in range(10)]; fr = src.read(32); print(fr.shape)
det = TRTYolo("models/yolo26m_384x640_b64_fp16.engine", conf=0.25)
x = torch.empty(32,3,384,640, dtype=torch.float16, device="cuda")
r, off = letterbox_into(x, fr)
d = det(x); b = unletterbox_boxes(d.boxes[0][d.valid[0]], r, off, src.hw)
print("trt", d.valid.sum(1).tolist()[:8]); print(torch.cat([b, d.scores[0][d.valid[0]][:,None], d.cls[0][d.valid[0]][:,None].float()],1)[:6].cpu().numpy().round(1))
tm = TorchYolo("models/yolo26m.pt", conf=0.25)
d2 = tm(x); print("torch", d2.valid.sum(1).tolist()[:8])
from ultralytics import YOLO
y = YOLO("models/yolo26m.pt"); img = fr[0].permute(1,2,0).cpu().numpy()[:,:,::-1].copy()
res = y.predict(img, imgsz=(384,640), conf=0.25, verbose=False)[0]
print("ultra", len(res.boxes)); print(torch.cat([res.boxes.xyxy, res.boxes.conf[:,None], res.boxes.cls[:,None]],1)[:6].cpu().numpy().round(1))
for B in ():
    xb = torch.randn(B,3,384,640,device="cuda").half()
    for _ in range(5): det.raw(xb)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(20): det.raw(xb)
    torch.cuda.synchronize(); dt=(time.time()-t)/20; 
    t=time.time()
    for _ in range(20): det(xb)
    torch.cuda.synchronize(); dt2=(time.time()-t)/20
    print(f"B={B} trt raw {dt*1e3:.2f} ms ({B/dt:.0f} img/s) +nms {dt2*1e3:.2f} ms")
