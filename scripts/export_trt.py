"""Export an Ultralytics YOLO model to a TensorRT FP16 engine with dynamic batch.

usage: python scripts/export_trt.py yolo26m.pt --imgsz 384 640 --batch 64
Writes models/<name>_<h>x<w>_b<batch>_fp16.engine (+ .json metadata).
"""
import argparse, json, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", default="yolo26m.pt", nargs="?")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 640])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--nms-free", action="store_true", help="YOLO26 one-to-one head: [B,max_det,6], no NMS")
    ap.add_argument("--max-det", type=int, default=150)
    a = ap.parse_args()
    from ultralytics import YOLO

    (ROOT / "models").mkdir(exist_ok=True)
    wpath = ROOT / "models" / a.weights
    if not wpath.exists():
        m = YOLO(a.weights)  # downloads into cwd
        shutil.move(a.weights, wpath)
    m = YOLO(str(wpath))
    t = time.time()
    out = m.export(format="engine", half=True, dynamic=True, batch=a.batch, imgsz=a.imgsz,
                   device=a.device, workspace=8, simplify=True,
                   **({"nms": False, "max_det": a.max_det} if a.nms_free else {}))
    h, w = a.imgsz
    tag = "_e2e" if a.nms_free else ""
    dst = ROOT / "models" / f"{wpath.stem}_{h}x{w}_b{a.batch}{tag}_fp16.engine"
    shutil.move(out, dst)
    meta = {"weights": a.weights, "imgsz": [h, w], "max_batch": a.batch, "fp16": True,
            "names": m.names, "end2end": bool(a.nms_free), "max_det": a.max_det,
            "export_s": round(time.time() - t, 1)}
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("engine:", dst, meta["end2end"], meta["export_s"], "s")


if __name__ == "__main__":
    main()
