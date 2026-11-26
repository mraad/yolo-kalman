"""Track one or more clips (as parallel streams on one GPU); write overlays, MOT txt and stats.

usage: ./run.sh scripts/track.py clips/*.webm --out results/real --overlay
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from yolo_kalman.overlay import OverlayWriter
from yolo_kalman.pipeline import Pipeline, PipelineConfig
from yolo_kalman.summary import track_stats, write_mot
from yolo_kalman.tracker import TrackerConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--engine", default="models/yolo26m_384x640_b64_e2e_fp16.engine")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--no-reid", action="store_true")
    ap.add_argument("--lost-s", type=float, default=8.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3, 5, 7, 14, 15, 16],
                    help="COCO ids: person, bicycle, car, motorcycle, bus, truck, bird, cat, dog")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(engine=a.engine, chunk=a.chunk, use_reid=not a.no_reid, max_frames=a.max_frames,
                         classes=a.classes, tracker=TrackerConfig(lost_buffer_s=a.lost_s))
    pipe = Pipeline(a.clips, gpu=a.gpu, cfg=cfg)
    writers = []
    if a.overlay:
        for p, s in zip(a.clips, pipe.srcs):
            writers.append(OverlayWriter(out / (Path(p).stem + "_tracks.mp4"), s.fps, s.hw, pipe.det.names))

        def on_chunk(frames, results, n):
            for i, f in enumerate(frames):
                if f is None:
                    continue
                imgs = f.permute(0, 2, 3, 1).cpu().numpy()
                trs = results[i].tracks[-int(n[i]):]
                for img, (ids, b, s, c) in zip(imgs, trs):
                    writers[i].write(img, ids, b, s, c)
        pipe.on_chunk = on_chunk
    t = time.time()
    res = pipe.run()
    wall = time.time() - t
    for w in writers:
        w.close()
    summary = {}
    for r in res:
        st = track_stats(r.tracks, r.fps)
        summary[Path(r.path).name] = st
        write_mot(r.tracks, out / (Path(r.path).stem + ".mot.txt"))
        print(f"{Path(r.path).name[:50]:50s} frames={st['frames']:5d} ids={st['unique_ids']:4d} "
              f"ids>=1s={st['ids_ge_1s']:4d} frag={st['fragmentations']:4d} reacq>=1s={st['reappear_after_gap_ge_1s']:3d}")
    summary["_run"] = {"wall_s": wall, "frames": pipe.frames_done, "fps": pipe.frames_done / wall,
                       "timing_s": pipe.timing, "tracker_timing_s": pipe.tracker.timing,
                       "tracker_stats": pipe.tracker.stats, "overlay": a.overlay}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["_run"], indent=1))


if __name__ == "__main__":
    main()
