"""End-to-end: NVDEC -> TensorRT YOLO -> re-ID -> tracker on real and rendered clips."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "models/yolo26m_384x640_b64_e2e_fp16.engine"
REAL = ROOT / "clips/London_United_LTZ1130_passing_House_of_Commons_in_May_2015.webm"
SYN = ROOT / "data/synth/synth_gap030.mp4"

pytestmark = pytest.mark.skipif(not ENGINE.exists(), reason="TensorRT engine not exported")


def _pipe(paths, **kw):
    from yolo_kalman.pipeline import Pipeline, PipelineConfig
    from yolo_kalman.tracker import TrackerConfig

    cfg = PipelineConfig(engine=str(ENGINE), tracker=TrackerConfig(compile=False), **kw)
    return Pipeline([str(p) for p in paths], gpu=0, cfg=cfg)


@pytest.mark.skipif(not REAL.exists(), reason="real clip not downloaded")
def test_real_clip_runs_and_tracks_bus():
    from yolo_kalman.summary import track_stats

    pipe = _pipe([REAL], classes=[0, 1, 2, 3, 5, 7])
    res = pipe.run()[0]
    assert res.frames == pipe.srcs[0].num_frames > 100
    st = track_stats(res.tracks, res.fps)
    assert st["unique_ids"] >= 1
    # the bus crossing the frame is followed by one id for most of the clip
    counts = {}
    for ids, b, s, c in res.tracks:
        for i, cc in zip(ids, c):
            if cc in (2, 5, 7):
                counts[int(i)] = counts.get(int(i), 0) + 1
    assert max(counts.values()) >= 0.6 * res.frames, counts
    assert pipe.timing["detect"] > 0 and pipe.tracker.timing["cpu_match"] > 0


@pytest.mark.skipif(not SYN.exists(), reason="run scripts/make_synthetic.py first")
def test_synthetic_clip_same_id_after_full_occlusion():
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_synth import load_gt, occlusion_check

    meta = json.loads((SYN.parent / "synth_gap030.json").read_text())
    pipe = _pipe([SYN], classes=[0])
    res = pipe.run()[0]
    gt = load_gt("synth_gap030", meta["length"], 0.3)
    tr = [(ids, b) for ids, b, *_ in res.tracks]
    occ = occlusion_check(gt, tr, meta["fully_hidden_frames"])
    assert occ and all(o["same"] for o in occ.values()), occ
