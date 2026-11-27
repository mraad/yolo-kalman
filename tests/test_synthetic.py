"""Synthetic scenes: identity must survive full occlusion gaps and crossings (no swaps)."""
import numpy as np
import pytest

from yolo_kalman.metrics import clear_id, hota, id_consistency
from yolo_kalman.simrun import run_scenes
from yolo_kalman.synthetic import Obj, Scene, crossing_scene, gap_scene
from yolo_kalman.tracker import TrackerConfig

GAPS = [10, 30, 60, 120]


def _check_same_ids(gt, out, n_obj):
    cons = id_consistency(gt, out)
    assert len(cons) == n_obj, cons
    for oid, tids in cons.items():
        assert len(tids) == 1, f"GT {oid} got ids {tids}"
    all_ids = [next(iter(v)) for v in cons.values()]
    assert len(set(all_ids)) == n_obj, f"ids shared between objects: {cons}"


@pytest.mark.parametrize("use_emb", [True, False], ids=["reid", "motion-only"])
def test_full_occlusion_gaps_batched(use_emb):
    """All four gap lengths run as four parallel streams of one tracker."""
    scenes = [gap_scene(g) for g in GAPS]
    gts, outs, trk = run_scenes(scenes, use_emb=use_emb)
    for g, gt, out, sc in zip(GAPS, gts, outs, scenes):
        _check_same_ids(gt, out, len(sc.objs))
        m = clear_id(gt, out)
        assert m["num_switches"] == 0, (g, m)
    assert trk.stats["oru"] > 0


def test_gap_with_speed_change_needs_reid():
    """Object changes speed while hidden (motion model is wrong) -> appearance re-ID recovers it."""
    sc = gap_scene(60, n_lanes=1)
    o = sc.objs[0]
    # piecewise: slows to half speed behind the bar -> reappears later than predicted
    slow = Obj(1, o.box0, o.vel, cls=0)
    import types

    bar_x0 = sc.bars[0][0]

    def box(self, t, _o=o):
        b = _o.box(t)
        x_enter = bar_x0 + 20
        if b[0] > x_enter:
            b[0] = x_enter + (b[0] - x_enter) * 0.6
        return b

    slow.box = types.MethodType(box, slow)
    sc.objs = [slow]
    sc.length += 60
    gts, outs, trk = run_scenes([sc], use_emb=True)
    _check_same_ids(gts[0], outs[0], 1)


def test_crossings_no_swap():
    scenes = [crossing_scene()]
    gts, outs, _ = run_scenes(scenes, use_emb=True)
    _check_same_ids(gts[0], outs[0], 4)
    gts2, outs2, _ = run_scenes(scenes, use_emb=False)
    _check_same_ids(gts2[0], outs2[0], 4)


def test_clutter_and_metrics():
    """Gap scenes with false positives: HOTA/IDF1 stay high, no id switches."""
    scenes = [gap_scene(g) for g in GAPS]
    gts, outs, _ = run_scenes(scenes, use_emb=True, fp_rate=0.5)
    for gt, out in zip(gts, outs):
        h = hota(gt, out)
        m = clear_id(gt, out)
        assert m["num_switches"] == 0, m
        assert m["idf1"] > 0.85, m
        assert h["HOTA"] > 0.7, h


@pytest.mark.parametrize("assign", ["gpu_hungarian", "lapjv"])
def test_cuda_graph_compiled_matches_eager(assign):
    """torch.compile + CUDA graph path gives the same ids/boxes as plain eager torch."""
    scenes = [gap_scene(30), crossing_scene()]
    _, outs_g, _ = run_scenes(scenes, cfg=TrackerConfig(cuda_graph=True, compile=True, assign=assign))
    _, outs_e, _ = run_scenes(scenes, cfg=TrackerConfig(cuda_graph=False, compile=False, assign=assign))
    for og, oe in zip(outs_g, outs_e):
        for (ig, bg), (ie, be) in zip(og, oe):
            np.testing.assert_array_equal(ig, ie)
            np.testing.assert_allclose(bg, be, atol=1e-3)


@pytest.mark.parametrize("use_emb", [True, False], ids=["reid", "motion-only"])
def test_gpu_hungarian_cascade_matches_lapjv(use_emb):
    """GPU cascade (batched Hungarian) == CPU cascade (lap.lapjv): same ids, boxes and counters."""
    scenes = [gap_scene(g) for g in GAPS] + [crossing_scene()]
    res = {}
    for a in ("lapjv", "gpu_hungarian"):
        _, outs, trk = run_scenes(scenes, cfg=TrackerConfig(compile=False, assign=a), use_emb=use_emb, fp_rate=0.5)
        res[a] = (outs, dict(trk.stats))
    (oa, sa), (ob, sb) = res["lapjv"], res["gpu_hungarian"]
    assert sa == sb
    for xa, xb in zip(oa, ob):
        assert len(xa) == len(xb)
        for (ia, ba), (ib, bb) in zip(xa, xb):
            np.testing.assert_array_equal(ia, ib)
            np.testing.assert_allclose(ba, bb, atol=1e-3)


def test_cpu_tracker_runs_lapjv_cascade():
    """A CPU tracker (lap_batch is CUDA-only) falls back to the lapjv cascade and matches it on the GPU."""
    scenes = [gap_scene(30), crossing_scene()]
    _, out_cpu, trk = run_scenes(scenes, cfg=TrackerConfig(compile=False), device="cpu", fp_rate=0.5)
    assert not trk.gpu_assign
    _, out_gpu, _ = run_scenes(scenes, cfg=TrackerConfig(compile=False, assign="lapjv"), fp_rate=0.5)
    for xa, xb in zip(out_cpu, out_gpu):
        assert len(xa) == len(xb)
        for (ia, ba), (ib, bb) in zip(xa, xb):
            np.testing.assert_array_equal(ia, ib)
            np.testing.assert_allclose(ba, bb, atol=1e-2)
