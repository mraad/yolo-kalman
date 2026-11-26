"""Synthetic scenes with known ground truth: moving boxes, crossings, full occlusion by grey bars.

Used by the unit tests (simulated detections + embeddings, no detector) and to render synthetic
occlusion clips for end-to-end evaluation with the real detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Obj:
    oid: int
    box0: tuple[float, float, float, float]   # cx, cy, w, h at frame start
    vel: tuple[float, float]                  # px / frame
    start: int = 0
    end: int = 10 ** 9
    cls: int = 0
    sprite: int = 0                           # index of sprite when rendering
    depth: float = 0.0                        # larger = closer to camera (occludes others)

    def box(self, t: int) -> np.ndarray:
        cx, cy, w, h = self.box0
        dt = t - self.start
        return np.array([cx + self.vel[0] * dt, cy + self.vel[1] * dt, w, h], np.float64)


@dataclass
class Scene:
    width: int
    height: int
    length: int
    objs: list[Obj]
    bars: list[tuple[float, float]] = field(default_factory=list)   # (x0, x1) full-height occluders
    fps: float = 30.0

    # ---- geometry --------------------------------------------------------------------------
    def gt(self):
        """Per frame list of (oid, xyxy, visibility, cls)."""
        out = []
        for t in range(self.length):
            alive = [o for o in self.objs if o.start <= t < o.end]
            frame = []
            for o in alive:
                b = _xyxy(o.box(t))
                if b[2] < 0 or b[0] > self.width or b[3] < 0 or b[1] > self.height:
                    continue
                vis = self._visibility(b, o, alive, t)
                frame.append((o.oid, b, vis, o.cls))
            out.append(frame)
        return out

    def _visibility(self, b, o, alive, t, n=24):
        xs = np.linspace(b[0], b[2], n)
        ys = np.linspace(b[1], b[3], n)
        X, Y = np.meshgrid(xs, ys)
        vis = (X >= 0) & (X <= self.width) & (Y >= 0) & (Y <= self.height)
        for x0, x1 in self.bars:
            vis &= ~((X >= x0) & (X <= x1))
        for p in alive:
            if p is o or p.depth <= o.depth:
                continue
            q = _xyxy(p.box(t))
            vis &= ~((X >= q[0]) & (X <= q[2]) & (Y >= q[1]) & (Y <= q[3]))
        return float(vis.mean())

    # ---- simulated detector ----------------------------------------------------------------
    def detections(self, seed=0, emb_dim=64, noise_px=1.0, fp_rate=0.0, emb_noise=0.15):
        """Per frame: (boxes [n,4], scores [n], cls [n], emb [n,D], oids [n]) mimicking a detector.

        visibility >= 0.6 -> full box, high score; 0.3..0.6 -> low-score box of the visible region
        (BYTE second stage); < 0.3 -> missed.
        """
        rng = np.random.default_rng(seed)
        base = {o.oid: _unit(rng.normal(size=emb_dim)) for o in self.objs}
        out = []
        for frame in self.gt():
            B, S, C, E, I = [], [], [], [], []
            for oid, b, vis, c in frame:
                if vis < 0.3:
                    continue
                bb = b + rng.normal(0, noise_px, 4)
                if vis >= 0.6:
                    s = 0.55 + 0.4 * vis + rng.uniform(-0.05, 0.04)
                else:
                    s = 0.15 + 0.4 * (vis - 0.3) / 0.3
                    bb = self._visible_part(bb, oid)
                B.append(bb); S.append(min(s, 0.99)); C.append(c)
                E.append(_unit(base[oid] + rng.normal(0, emb_noise / np.sqrt(emb_dim), emb_dim)))
                I.append(oid)
            for _ in range(rng.poisson(fp_rate)):
                cx, cy = rng.uniform(0, self.width), rng.uniform(0, self.height)
                B.append(np.array([cx - 15, cy - 30, cx + 15, cy + 30])); S.append(rng.uniform(0.1, 0.55)); C.append(0)
                E.append(_unit(rng.normal(size=emb_dim))); I.append(-1)
            out.append((np.array(B, np.float32).reshape(-1, 4), np.array(S, np.float32), np.array(C, np.int64),
                        np.array(E, np.float32).reshape(-1, emb_dim), np.array(I, np.int64)))
        return out

    def _visible_part(self, b, oid):
        x0, y0, x1, y1 = b
        for bx0, bx1 in self.bars:
            if bx0 <= x0 <= bx1:
                x0 = bx1
            if bx0 <= x1 <= bx1:
                x1 = bx0
        if x1 - x0 < 2:
            return b
        return np.array([x0, y0, x1, y1])

    # ---- rendering ---------------------------------------------------------------------------
    def render(self, sprites: list[np.ndarray], bg: np.ndarray | None = None, bar_color=(128, 128, 128)):
        """Yield RGB uint8 frames. sprites: list of RGBA/RGB arrays, resized to each object's box."""
        import cv2

        if bg is None:
            yy, xx = np.mgrid[0:self.height, 0:self.width]
            bg = np.stack([60 + 40 * xx / self.width, 90 + 30 * yy / self.height, 70 + 0 * xx], -1)
            bg = (bg + np.random.default_rng(0).normal(0, 4, bg.shape)).clip(0, 255).astype(np.uint8)
        cache = {}
        for t in range(self.length):
            img = bg.copy()
            alive = sorted([o for o in self.objs if o.start <= t < o.end], key=lambda o: o.depth)
            for o in alive:
                x0, y0, x1, y1 = np.round(_xyxy(o.box(t))).astype(int)
                w, h = x1 - x0, y1 - y0
                key = (o.sprite, w, h)
                if key not in cache:
                    cache[key] = cv2.resize(sprites[o.sprite], (w, h), interpolation=cv2.INTER_AREA)
                sp = cache[key]
                cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, self.width), min(y1, self.height)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                patch = sp[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
                if patch.shape[2] == 4:
                    a = patch[..., 3:4].astype(np.float32) / 255
                    img[cy0:cy1, cx0:cx1] = (a * patch[..., :3] + (1 - a) * img[cy0:cy1, cx0:cx1]).astype(np.uint8)
                else:
                    img[cy0:cy1, cx0:cx1] = patch
            for bx0, bx1 in self.bars:
                img[:, int(bx0):int(np.ceil(bx1))] = bar_color
            yield img


def _xyxy(b):
    cx, cy, w, h = b
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


# ---- canned scenes ----------------------------------------------------------------------------
def gap_scene(gap: int, speed: float = 3.0, w: float = 40, h: float = 90, width: int = 1280, height: int = 480,
              n_lanes: int = 3, fps: float = 30.0) -> Scene:
    """Objects in lanes walking right behind a grey bar that hides each fully for ``gap`` frames."""
    bar_w = w + speed * gap
    x0 = 200.0
    bar = (x0, x0 + bar_w)
    objs = []
    lead = 60
    for i in range(n_lanes):
        cy = 80 + i * (height - 160) / max(1, n_lanes - 1)
        sp = speed
        start_x = x0 - w / 2 - sp * lead - 40 * i
        objs.append(Obj(i + 1, (start_x, cy, w, h), (sp, 0.0), cls=0, sprite=i))
    # frames: enter, hide for gap, reappear, then 60 more frames
    length = int(lead + 40 + gap + (w + 40 * n_lanes) / speed + 60)
    width = int(max(width, bar[1] + w * 2 + speed * 80))
    return Scene(width, height, length, objs, [bar], fps)


def crossing_scene(width=960, height=480, length=200, fps=30.0) -> Scene:
    """Two pairs of objects crossing: horizontal head-on crossing (one passes in front of the
    other, fully occluding it for a few frames) and a diagonal X crossing."""
    objs = [
        Obj(1, (100, 200, 50, 110), (4.0, 0.0), cls=0, sprite=0, depth=1.0),
        Obj(2, (860, 205, 50, 110), (-4.0, 0.0), cls=0, sprite=1, depth=0.0),
        Obj(3, (150, 60, 45, 100), (3.5, 1.4), cls=0, sprite=2, depth=0.5),
        Obj(4, (810, 60, 45, 100), (-3.5, 1.4), cls=0, sprite=3, depth=0.2),
    ]
    return Scene(width, height, length, objs, [], fps)
