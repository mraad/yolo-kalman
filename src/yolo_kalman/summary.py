"""Per-clip tracking statistics without ground truth."""
from __future__ import annotations

import numpy as np


def track_stats(tracks, fps: float, reacq_min_gap: int = 5):
    """tracks: per frame (ids, boxes, scores, cls).

    fragmentations = number of times a track id disappears and later reappears;
    reacquired_long = reappearances after a gap >= 1 s (the occlusion case we care about).
    """
    seen: dict[int, list[int]] = {}
    for t, (ids, *_r) in enumerate(tracks):
        for i in ids:
            seen.setdefault(int(i), []).append(t)
    frags, long_re, lengths = 0, 0, []
    gaps = []
    for i, fr in seen.items():
        fr = np.array(fr)
        lengths.append(len(fr))
        d = np.diff(fr)
        g = d[d > 1] - 1
        frags += int((g >= 1).sum())
        long_re += int((g >= fps).sum())
        gaps.extend(g.tolist())
    lengths = np.array(lengths) if lengths else np.zeros(1)
    return {
        "frames": len(tracks),
        "unique_ids": len(seen),
        "ids_ge_1s": int((lengths >= fps).sum()),
        "mean_track_len_frames": float(lengths.mean()),
        "fragmentations": frags,
        "reappear_after_gap_ge_1s": long_re,
        "max_gap_bridged_frames": int(max(gaps) if gaps else 0),
        "mean_boxes_per_frame": float(np.mean([len(x[0]) for x in tracks])) if tracks else 0.0,
    }


def write_mot(tracks, path):
    with open(path, "w") as f:
        for t, (ids, boxes, scores, cls) in enumerate(tracks):
            for i, b, s, c in zip(ids, boxes, scores, cls):
                f.write(f"{t + 1},{int(i)},{b[0]:.1f},{b[1]:.1f},{b[2] - b[0]:.1f},{b[3] - b[1]:.1f},{s:.3f},{int(c)},-1,-1\n")
