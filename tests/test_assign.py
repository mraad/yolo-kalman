"""Batched GPU Hungarian (assign.lap_batch) vs lap.lapjv / scipy linear_sum_assignment: equal total cost."""
import lap
import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from yolo_kalman.assign import extended_cost, lap_batch, lap_batch_torch

DEV = "cuda"
SOLVERS = {"cuda": lap_batch, "torch": lap_batch_torch}


def scipy_opt(sub: np.ndarray, limit: float) -> float:
    """Optimum of lap's extend_cost/cost_limit problem via scipy on the explicit extended matrix."""
    n, m = sub.shape
    if n == 0 or m == 0:
        return limit / 2 * (n + m)
    e = np.full((n + m, n + m), limit / 2)
    e[n:, m:] = 0
    e[:n, :m] = sub
    r, k = linear_sum_assignment(e)
    return float(e[r, k].sum())


def lapjv_cost(sub: np.ndarray, limit: float) -> float:
    if sub.size == 0:
        return limit / 2 * sum(sub.shape)
    _, x, _ = lap.lapjv(sub, extend_cost=True, cost_limit=limit)
    r = np.nonzero(x >= 0)[0]
    return extended_cost(sub, r, x[r], limit)


def check_batch(cost, rm, cm, limit, solver):
    out = SOLVERS[solver](torch.from_numpy(cost).to(DEV), torch.from_numpy(rm).to(DEV),
                          torch.from_numpy(cm).to(DEV), limit).cpu().numpy()
    assert out.shape == rm.shape
    for b in range(cost.shape[0]):
        o = out[b]
        rows, cols = np.nonzero(rm[b])[0], np.nonzero(cm[b])[0]
        assert (o[~rm[b]] == -1).all()
        mr = np.nonzero(o >= 0)[0]
        assert cm[b][o[mr]].all(), "matched a masked column"
        assert len(set(o[mr].tolist())) == len(mr), "column used twice"
        sub = cost[b][np.ix_(rows, cols)].astype(np.float64)
        assert (cost[b][mr, o[mr]] <= limit).all(), "matched a pair above cost_limit"
        got = extended_cost(sub, np.searchsorted(rows, mr), np.searchsorted(cols, o[mr]), limit)
        want = scipy_opt(sub, limit)
        assert got == pytest.approx(want, abs=1e-6), (b, got, want)
        assert lapjv_cost(sub, limit) == pytest.approx(want, abs=1e-6)
    return out


def random_batch(rng, B, R, C, gate_p=0.5, ties=False, big=6e4, mask_p=0.8):
    cost = rng.random((B, R, C)).astype(np.float32)
    if ties:
        cost = np.round(cost * 4).astype(np.float32) / 4
    cost[rng.random(cost.shape) < gate_p] = big
    rm = rng.random((B, R)) < mask_p
    cm = rng.random((B, C)) < mask_p
    return cost, rm, cm


@pytest.mark.parametrize("solver", ["cuda", "torch"])
def test_random_rectangular(solver):
    rng = np.random.default_rng(1)
    n_trials = 150 if solver == "cuda" else 40
    for t in range(n_trials):
        R, C = int(rng.integers(1, 48)), int(rng.integers(1, 48))
        cost, rm, cm = random_batch(rng, 6, R, C, gate_p=rng.choice([0.0, 0.5, 0.9]), ties=t % 3 == 0)
        check_batch(cost, rm, cm, float(rng.choice([0.25, 0.5, 0.7, 0.8, 1.5])), solver)


@pytest.mark.parametrize("solver", ["cuda", "torch"])
def test_edge_cases(solver):
    rng = np.random.default_rng(2)
    lim = 0.8
    # 1xN, Nx1
    for R, C in [(1, 1), (1, 17), (17, 1), (1, 160), (160, 1)]:
        cost, rm, cm = random_batch(rng, 4, R, C, gate_p=0.3, mask_p=1.0)
        check_batch(cost, rm, cm, lim, solver)
    # empty: all rows / all cols masked, zero-size dims
    cost, rm, cm = random_batch(rng, 3, 10, 12)
    out = check_batch(cost, np.zeros_like(rm), cm, lim, solver)
    assert (out == -1).all()
    out = check_batch(cost, rm, np.zeros_like(cm), lim, solver)
    assert (out == -1).all()
    e = SOLVERS[solver](torch.zeros(3, 5, 0, device=DEV), torch.ones(3, 5, dtype=torch.bool, device=DEV),
                        torch.ones(3, 0, dtype=torch.bool, device=DEV), lim)
    assert e.shape == (3, 5) and (e == -1).all()
    e = SOLVERS[solver](torch.zeros(3, 0, 5, device=DEV), torch.ones(3, 0, dtype=torch.bool, device=DEV),
                        torch.ones(3, 5, dtype=torch.bool, device=DEV), lim)
    assert e.shape == (3, 0)
    # all gated (every cost above the limit)
    cost = np.full((4, 9, 7), 6e4, np.float32)
    out = check_batch(cost, np.ones((4, 9), bool), np.ones((4, 7), bool), lim, solver)
    assert (out == -1).all()
    cost = (lim + 1e-3 + rng.random((4, 9, 7))).astype(np.float32)
    assert (check_batch(cost, np.ones((4, 9), bool), np.ones((4, 7), bool), lim, solver) == -1).all()
    # large values (fp16 max, huge) mixed with small ones; non-finite entries never matched
    cost, rm, cm = random_batch(rng, 4, 20, 25, gate_p=0.4, big=3.0e38)
    check_batch(cost, rm, cm, 5e4, solver)
    cost, rm, cm = random_batch(rng, 4, 20, 25, gate_p=0.4, big=6.55e4)
    cost[0, 0, :] = np.inf
    cost[1, :, 0] = np.nan
    out = SOLVERS[solver](torch.from_numpy(cost).to(DEV), torch.from_numpy(rm).to(DEV),
                          torch.from_numpy(cm).to(DEV), lim).cpu().numpy()
    assert out[0, 0] == -1 and (out[1] != 0).all()
    # values exactly at the limit: total still optimal (tie with "unmatched")
    cost = np.where(rng.random((4, 8, 8)) < 0.5, lim, 6e4).astype(np.float32)
    check_batch(cost, np.ones((4, 8), bool), np.ones((4, 8), bool), lim, solver)
    # integer costs, large magnitude, many ties
    cost = rng.integers(0, 5, (4, 30, 20)).astype(np.float32) * 1000.0
    check_batch(cost, np.ones((4, 30), bool), np.ones((4, 20), bool), 2500.0, solver)


def test_ties_deterministic():
    rng = np.random.default_rng(3)
    cost = np.round(rng.random((8, 40, 40)) * 2).astype(np.float32) / 2   # 0, 0.5, 1 -> many optima
    rm, cm = np.ones((8, 40), bool), np.ones((8, 40), bool)
    outs = [check_batch(cost, rm, cm, 0.8, "cuda") for _ in range(3)]
    assert all((o == outs[0]).all() for o in outs)
    # identical problems in every batch element -> identical answers
    same = np.repeat(cost[:1], 8, 0)
    o = check_batch(same, rm, cm, 0.8, "cuda")
    assert (o == o[:1]).all()


def test_batched_mixed_sizes():
    """Streams with different live sizes padded into one [S, M, N] batch (the tracker layout)."""
    rng = np.random.default_rng(4)
    S, M, N = 32, 160, 160
    cost = np.full((S, M, N), 6e4, np.float32)
    rm = np.zeros((S, M), bool)
    cm = np.zeros((S, N), bool)
    for s in range(S):
        n, m = int(rng.integers(0, 80)), int(rng.integers(0, 80))
        rows = np.sort(rng.choice(M, n, replace=False))
        cols = np.sort(rng.choice(N, m, replace=False))
        rm[s, rows] = True
        cm[s, cols] = True
        sub = rng.random((n, m)).astype(np.float32)
        sub[rng.random((n, m)) < 0.7] = 6e4
        cost[s][np.ix_(rows, cols)] = sub
    check_batch(cost, rm, cm, 0.8, "cuda")
    # torch implementation gives the same assignment on a tie-free problem
    a = lap_batch(torch.from_numpy(cost).to(DEV), torch.from_numpy(rm).to(DEV), torch.from_numpy(cm).to(DEV), 0.8)
    b = lap_batch_torch(torch.from_numpy(cost).to(DEV), torch.from_numpy(rm).to(DEV),
                        torch.from_numpy(cm).to(DEV), 0.8)
    assert torch.equal(a, b)


def test_cuda_graph_capture_and_compile():
    """The op is one fixed launch: capturable in a CUDA graph and traceable by torch.compile."""
    rng = np.random.default_rng(5)
    cost, rm, cm = random_batch(rng, 16, 40, 30)
    tc, trm, tcm = (torch.from_numpy(x).to(DEV) for x in (cost, rm, cm))
    ref = lap_batch(tc, trm, tcm, 0.8)
    f = torch.compile(lambda c, r, k: lap_batch(c * 1.0, r, k, 0.8), dynamic=False)
    assert torch.equal(f(tc, trm, tcm), ref)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        lap_batch(tc, trm, tcm, 0.8)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = lap_batch(tc, trm, tcm, 0.8)
    cost2, rm2, cm2 = random_batch(rng, 16, 40, 30)
    tc.copy_(torch.from_numpy(cost2))
    trm.copy_(torch.from_numpy(rm2))
    tcm.copy_(torch.from_numpy(cm2))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, lap_batch(tc, trm, tcm, 0.8))
    check_batch(cost2, rm2, cm2, 0.8, "cuda")
