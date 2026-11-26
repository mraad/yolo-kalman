"""Render synthetic occlusion clips with known ground truth.

Person sprites are cropped (with the detector) from Ultralytics' bundled sample images
(bus.jpg / zidane.jpg, AGPL-3.0).  Each clip: people walk behind a grey bar that hides them
completely for a known number of frames (10 / 30 / 60 / 120), plus a crossing clip where people
pass in front of each other.

Writes data/synth/<name>.mp4, <name>.gt.txt (MOT format: frame,id,x,y,w,h,1,1,visibility) and
<name>.json (scene definition incl. full-occlusion frame ranges).
"""
import json
from pathlib import Path

import cv2
import numpy as np

from yolo_kalman.synthetic import Obj, Scene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/synth"


def person_sprites():
    import ultralytics
    from ultralytics import YOLO

    assets = Path(ultralytics.__file__).parent / "assets"
    model = YOLO(str(ROOT / "models/yolo26m.pt"))
    sprites = []
    for name in ("bus.jpg", "zidane.jpg"):
        img = cv2.imread(str(assets / name))[:, :, ::-1]
        r = model.predict(img, conf=0.6, classes=[0], verbose=False)[0]
        for b in r.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = b
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] > 100 and crop.shape[1] > 40:
                sprites.append(np.ascontiguousarray(crop))
    return sprites


def gap_clip(gap, sprite_ids, W=854, H=480, speed=None, w=56, h=140, fps=30.0):
    speed = speed or (2.5 if gap <= 60 else 2.0)
    bar_w = w + speed * gap
    x0 = 230.0
    objs = []
    lanes = [(140, 0), (350, 1)]
    for k, (cy, lane) in enumerate(lanes):
        start_x = -w / 2 + 10 + 0 * k
        objs.append(Obj(k + 1, (start_x - 70 * k, cy, w, h), (speed, 0.0), sprite=sprite_ids[k]))
    # a third person walking left in the lower lane, crossing the bar later
    objs.append(Obj(3, (W + w / 2 + 40, 250, w * 0.9, h * 0.9), (-speed * 0.8, 0.0), start=0,
                    sprite=sprite_ids[2]))
    length = int((W + 2 * w + 70) / (speed * 0.8)) + 10
    return Scene(W, H, length, objs, [(x0, x0 + bar_w)], fps)


def crossing_clip(sprite_ids, W=854, H=480, fps=30.0):
    """Head-on horizontal crossing (A passes in front of B and hides it completely) in the lower
    half, and an X-shaped diagonal crossing in the upper half."""
    objs = [
        Obj(1, (40, 340, 72, 175), (2.0, 0.0), sprite=sprite_ids[0], depth=1.0),
        Obj(2, (814, 335, 48, 120), (-1.5, 0.0), sprite=sprite_ids[1], depth=0.0),
        Obj(3, (90, 70, 44, 110), (2.6, 0.45), sprite=sprite_ids[2], depth=0.5),
        Obj(4, (764, 70, 44, 110), (-2.6, 0.45), sprite=sprite_ids[3], depth=0.2),
    ]
    return Scene(W, H, 330, objs, [], fps)


def write(name, scene, sprites):
    OUT.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(OUT / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), scene.fps,
                         (scene.width, scene.height))
    for img in scene.render(sprites):
        vw.write(img[:, :, ::-1])
    vw.release()
    hidden = {}
    with open(OUT / f"{name}.gt.txt", "w") as f:
        for t, frame in enumerate(scene.gt()):
            for oid, b, vis, c in frame:
                f.write(f"{t + 1},{oid},{b[0]:.1f},{b[1]:.1f},{b[2] - b[0]:.1f},{b[3] - b[1]:.1f},1,1,{vis:.3f}\n")
                if vis < 0.05:  # (practically) fully hidden
                    hidden.setdefault(oid, []).append(t)
    ranges = {}  # oid -> list of [first, last, n] contiguous fully-hidden runs (>= 3 frames)
    for oid, v in hidden.items():
        runs, start = [], v[0]
        for a, b in zip(v, v[1:] + [None]):
            if b is None or b != a + 1:
                if a - start + 1 >= 3:
                    runs.append([start, a, a - start + 1])
                start = b
        if runs:
            ranges[oid] = runs
    meta = {"name": name, "width": scene.width, "height": scene.height, "length": scene.length,
            "fps": scene.fps, "bars": scene.bars, "fully_hidden_frames": ranges,
            "objects": [o.__dict__ for o in scene.objs]}
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
    print(name, scene.length, "frames; fully hidden:", ranges)


def main():
    sprites = person_sprites()
    print(len(sprites), "sprites", [s.shape for s in sprites])
    n = len(sprites)
    for gap in (10, 30, 60, 120):
        write(f"synth_gap{gap:03d}", gap_clip(gap, [0, 2, 5 % n]), sprites)
    write("synth_crossing", crossing_clip([0, 2, 5 % n, 3 % n]), sprites)


if __name__ == "__main__":
    main()
