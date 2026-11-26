"""Batched torch KF vs a straightforward NumPy reference (explicit F, H, Q, R matrices)."""
import numpy as np
import pytest
import torch

from yolo_kalman import kalman as kf

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SP, SV = 1 / 20, 1 / 160


def np_initiate(z):
    w, h = z[2], z[3]
    std = np.array([2 * SP * w, 2 * SP * h, 2 * SP * w, 2 * SP * h, 10 * SV * w, 10 * SV * h, 10 * SV * w, 10 * SV * h])
    return np.r_[z, np.zeros(4)], np.diag(std ** 2)


def np_predict(x, P):
    F = np.eye(8); F[:4, 4:] = np.eye(4)
    w, h = x[2], x[3]
    q = np.array([SP * w, SP * h, SP * w, SP * h, SV * w, SV * h, SV * w, SV * h]) ** 2
    return F @ x, F @ P @ F.T + np.diag(q)


def np_update(x, P, z, score=None):
    H = np.zeros((4, 8)); H[:, :4] = np.eye(4)
    w, h = x[2], x[3]
    r = np.array([SP * w, SP * h, SP * w, SP * h]) ** 2
    if score is not None:
        r = r * (1 - min(score, 0.99))
    S = H @ P @ H.T + np.diag(r)
    K = P @ H.T @ np.linalg.inv(S)
    x2 = x + K @ (z - H @ x)
    P2 = (np.eye(8) - K @ H) @ P
    return x2, P2, S


def test_kf_sequence_matches_numpy():
    rng = np.random.default_rng(0)
    B, T = 64, 40
    z0 = np.c_[rng.uniform(50, 500, (B, 2)), rng.uniform(10, 100, (B, 2))]
    vel = rng.normal(0, 3, (B, 4)); vel[:, 2:] *= 0.1
    zt = torch.tensor(z0, dtype=torch.float64, device=DEV)
    m, c = kf.initiate(zt)
    ref = [np_initiate(z0[b]) for b in range(B)]
    for t in range(1, T):
        m, c = kf.predict(m, c)
        ref = [np_predict(*r) for r in ref]
        z = z0 + vel * t + rng.normal(0, 1, (B, 4))
        sc = rng.uniform(0.1, 1.0, B)
        upd = rng.random(B) > 0.3  # some frames without measurement
        mu, cu = kf.update(m, c, torch.tensor(z, device=DEV), torch.tensor(sc, device=DEV))
        u = torch.tensor(upd, device=DEV)
        m = torch.where(u[:, None], mu, m); c = torch.where(u[:, None, None], cu, c)
        ref = [np_update(*ref[b], z[b], sc[b])[:2] if upd[b] else ref[b] for b in range(B)]
    rm = np.stack([r[0] for r in ref]); rc = np.stack([r[1] for r in ref])
    np.testing.assert_allclose(m.cpu().numpy(), rm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c.cpu().numpy(), rc, rtol=1e-6, atol=1e-6)


def test_kf_float32_close():
    rng = np.random.default_rng(1)
    z0 = np.c_[rng.uniform(50, 500, (16, 2)), rng.uniform(10, 100, (16, 2))]
    m64, c64 = kf.initiate(torch.tensor(z0, device=DEV))
    m32, c32 = kf.initiate(torch.tensor(z0, device=DEV, dtype=torch.float32))
    for t in range(30):
        m64, c64 = kf.predict(m64, c64); m32, c32 = kf.predict(m32, c32)
        z = torch.tensor(z0 + t, device=DEV)
        m64, c64 = kf.update(m64, c64, z); m32, c32 = kf.update(m32, c32, z.float())
    np.testing.assert_allclose(m32.cpu().numpy(), m64.cpu().numpy(), rtol=1e-4, atol=1e-2)


def test_mahalanobis_matches_numpy():
    rng = np.random.default_rng(2)
    M, N = 5, 7
    z0 = np.c_[rng.uniform(50, 500, (M, 2)), rng.uniform(10, 100, (M, 2))]
    m, c = kf.initiate(torch.tensor(z0, device=DEV))
    m, c = kf.predict(m, c)
    dets = np.c_[rng.uniform(50, 500, (N, 2)), rng.uniform(10, 100, (N, 2))]
    d2 = kf.mahalanobis_sq(m, c, torch.tensor(dets, device=DEV)).cpu().numpy()
    for i in range(M):
        x, P = m[i].cpu().numpy(), c[i].cpu().numpy()
        _, _, S = np_update(x, P, x[:4])
        for j in range(N):
            d = dets[j] - x[:4]
            assert d2[i, j] == pytest.approx(d @ np.linalg.inv(S) @ d, rel=1e-6)


def test_predict_dt_matches_repeated_predict_when_size_static():
    rng = np.random.default_rng(3)
    z0 = np.c_[rng.uniform(50, 500, (8, 2)), rng.uniform(10, 100, (8, 2))]
    m, c = kf.initiate(torch.tensor(z0, device=DEV))
    m[:, 4:6] = torch.tensor(rng.normal(0, 3, (8, 2)), device=DEV)  # moving, size velocity 0
    m1, c1 = kf.predict_dt(m, c, torch.ones(8, device=DEV, dtype=torch.float64))
    m1r, c1r = kf.predict(m, c)
    np.testing.assert_allclose(c1.cpu().numpy(), c1r.cpu().numpy(), rtol=1e-12)
    mr, cr = m, c
    for _ in range(7):
        mr, cr = kf.predict(mr, cr)
    m7, c7 = kf.predict_dt(m, c, torch.full((8,), 7.0, device=DEV, dtype=torch.float64))
    np.testing.assert_allclose(m7.cpu().numpy(), mr.cpu().numpy(), rtol=1e-12)
    np.testing.assert_allclose(c7.cpu().numpy(), cr.cpu().numpy(), rtol=1e-10)


@pytest.mark.parametrize("unrolled", [False, True])
def test_unrolled_4x4_matches_linalg(unrolled, monkeypatch):
    monkeypatch.setattr(kf, "UNROLLED_4X4", unrolled)
    rng = np.random.default_rng(4)
    A = rng.normal(size=(32, 4, 4))
    S = A @ A.transpose(0, 2, 1) + 4 * np.eye(4)
    B = rng.normal(size=(32, 4, 8))
    St, Bt = torch.tensor(S, device=DEV), torch.tensor(B, device=DEV)
    np.testing.assert_allclose(kf.spd_solve(St, Bt).cpu().numpy(), np.linalg.solve(S, B), rtol=1e-9, atol=1e-9)
    ref = np.einsum("bin,bij,bjn->bn", B[..., :4, :], np.linalg.inv(S), B[..., :4, :])
    np.testing.assert_allclose(kf.tri_solve_sq_norm(St, Bt).cpu().numpy(), ref, rtol=1e-9)
