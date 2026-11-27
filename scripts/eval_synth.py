"""Evaluate the full pipeline (real detector) on the rendered synthetic occlusion clips.

Metrics vs ground truth (objects with visibility >= --min-vis): HOTA / DetA / AssA (own TrackEval
re-implementation), IDF1 / MOTA / ID switches / fragmentations (motmetrics), and the direct check
"same id before and after each full occlusion".  Runs ablations of the tracker.

usage: ./run.sh scripts/eval_synth.py [--overlay]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from yolo_kalman.metrics import clear_id, hota, iou_xyxy
from yolo_kalman.pipeline import Pipeline, PipelineConfig
from yolo_kalman.tracker import TrackerConfig

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data/synth"
CLIPS = ["synth_gap010", "synth_gap030", "synth_gap060", "synth_gap120", "synth_crossing"]

VARIANTS = {
    "full (BYTE+OC-SORT+re-ID)": dict(),
    "no re-ID": dict(_reid=False),
    "no ORU/OCM/OCR": dict(use_oru=False, ocm_weight=0.0, ocr_max_gap=0),
    "ByteTrack-like (1 s buffer, no OC-SORT, no re-ID)": dict(_reid=False, use_oru=False, ocm_weight=0.0,
                                                               ocr_max_gap=0, lost_buffer_s=1.0),
}


def load_gt(name, length, min_vis):
    rows = np.loadtxt(SYN / f"{name}.gt.txt", delimiter=",").reshape(-1, 9)
    frames = []
    for t in range(1, length + 1):
        r = rows[(rows[:, 0] == t) & (rows[:, 8] >= min_vis)]
        b = r[:, 2:6].copy()
        b[:, 2:] += b[:, :2]
        frames.append((r[:, 1].astype(np.int64), b))
    return frames


def occlusion_check(gt, tr, hidden, iou_thr=0.5, window=15):
    """For each fully-hidden interval: id matched in the `window` frames before vs after."""
    def ids_in(rng, oid):
        got = []
        for t in rng:
            if t < 0 or t >= len(gt):
                continue
            gi, gb = gt[t]
            ti, tb = tr[t]
            if oid not in gi or not len(ti):
                continue
            S = iou_xyxy(gb, tb)
            r, c = linear_sum_assignment(-S)
            for a, b in zip(r, c):
                if gi[a] == oid and S[a, b] >= iou_thr:
                    got.append(int(ti[b]))
        return got

    out = {}
    for oid, runs in hidden.items():
        for t0, t1, n in runs:
            before = ids_in(range(t0 - window - 20, t0), int(oid))
            first = ids_in(range(t1 + 1, t1 + 60), int(oid))
            settled = ids_in(range(t1 + 30, t1 + 30 + window), int(oid))   # ~1 s after reappearing
            bid = max(set(before), key=before.count) if before else None
            sid = max(set(settled), key=settled.count) if settled else None
            out[f"{oid}@{t0}"] = {"hidden_frames": n, "id_before": bid, "id_first_after": first[0] if first else None,
                                  "id_1s_after": sid,
                                  # 1 s later (or first id if the object is not visible then)
                                  "same": bid is not None and bid == (sid if sid is not None else (first[0] if first else None)),
                                  "same_immediately": bid is not None and first[:1] == [bid]}
    return out


def run_variant(name, kw, overlay_dir=None, min_vis=0.3, assign="gpu_hungarian"):
    kw = dict(kw, assign=assign)
    reid = kw.pop("_reid", True)
    cfg = PipelineConfig(use_reid=reid, classes=[0], tracker=TrackerConfig(**kw))
    paths = [str(SYN / f"{c}.mp4") for c in CLIPS]
    pipe = Pipeline(paths, gpu=0, cfg=cfg)
    writers = []
    if overlay_dir:
        from yolo_kalman.overlay import OverlayWriter
        for c, s in zip(CLIPS, pipe.srcs):
            writers.append(OverlayWriter(Path(overlay_dir) / f"{c}_tracks.mp4", s.fps, s.hw, pipe.det.names))

        def on_chunk(frames, results, n):
            for i, f in enumerate(frames):
                if f is None:
                    continue
                for img, (ids, b, s, cl) in zip(f.permute(0, 2, 3, 1).cpu().numpy(), results[i].tracks[-int(n[i]):]):
                    writers[i].write(img, ids, b, s, cl)
        pipe.on_chunk = on_chunk
    res = pipe.run()
    for w in writers:
        w.close()
    rows = {}
    for c, r in zip(CLIPS, res):
        meta = json.loads((SYN / f"{c}.json").read_text())
        gt = load_gt(c, meta["length"], min_vis)
        tr = [(ids, b) for ids, b, *_ in r.tracks]
        tr += [(np.zeros(0, np.int64), np.zeros((0, 4)))] * (len(gt) - len(tr))
        h = hota(gt, tr)
        m = clear_id(gt, tr)
        occ = occlusion_check(gt, tr, meta["fully_hidden_frames"])
        rows[c] = {**h, **m, "occlusions": occ,
                   "occl_same_id": f"{sum(o['same'] for o in occ.values())}/{len(occ)}",
                   "occl_same_id_immediate": f"{sum(o['same_immediately'] for o in occ.values())}/{len(occ)}",
                   "pred_ids": len({int(i) for ids, _ in tr for i in ids})}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--min-vis", type=float, default=0.3)
    ap.add_argument("--assign", choices=["gpu_hungarian", "lapjv"], default="gpu_hungarian")
    a = ap.parse_args()
    sfx = "" if a.assign == "gpu_hungarian" else f"_{a.assign}"
    out = ROOT / "results/synth"
    out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for i, (vname, kw) in enumerate(VARIANTS.items()):
        allres[vname] = run_variant(vname, kw, out if (a.overlay and i == 0) else None, a.min_vis, a.assign)
    (out / f"metrics{sfx}.json").write_text(json.dumps(allres, indent=1, default=str))
    lines = ["| variant | clip | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | frag | GT ids | pred ids | same id after full occlusion (1 s later / immediately) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for vname, rows in allres.items():
        for c, r in rows.items():
            lines.append(f"| {vname} | {c} | {r['HOTA']:.3f} | {r['AssA']:.3f} | {r['DetA']:.3f} | {r['idf1']:.3f} | "
                         f"{r['mota']:.3f} | {int(r['num_switches'])} | {int(r['num_fragmentations'])} | "
                         f"{int(r['num_unique_objects'])} | {r['pred_ids']} | {r['occl_same_id']} / {r['occl_same_id_immediate']} |")
    (out / f"metrics{sfx}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
