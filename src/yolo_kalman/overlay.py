"""Overlay rendering (CPU, visualisation only): boxes + ids + short trails."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def id_color(i: int):
    rng = np.random.default_rng(int(i) * 7919 + 17)
    c = rng.integers(60, 255, 3)
    return int(c[0]), int(c[1]), int(c[2])


class OverlayWriter:
    def __init__(self, path: str | Path, fps: float, hw, names: dict[int, str] | None = None, trail: int = 30):
        h, w = hw
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self.names = names or {}
        self.trail = trail
        self.hist: dict[int, list] = {}
        self.t = 0

    def write(self, frame_rgb: np.ndarray, ids, boxes, scores, cls):
        img = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        for i, b, s, c in zip(ids, boxes, scores, cls):
            col = id_color(i)
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            label = f"{int(i)} {self.names.get(int(c), int(c))}"
            cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            h = self.hist.setdefault(int(i), [])
            h.append((self.t, (x1 + x2) // 2, y2))
        for i, h in list(self.hist.items()):
            pts = [(x, y) for t, x, y in h if self.t - t < self.trail]
            if not pts:
                del self.hist[i]
                continue
            self.hist[i] = [p for p in h if self.t - p[0] < self.trail]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(img, a, b, id_color(i), 1)
        cv2.putText(img, f"frame {self.t}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.vw.write(img)
        self.t += 1

    def close(self):
        self.vw.release()
