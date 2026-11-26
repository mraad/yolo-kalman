"""End-to-end throughput benchmark: N parallel streams split across GPUs (one worker process per GPU).

usage: ./run.sh scripts/bench.py --streams 1 4 8 16 --gpus 0 1 --frames 1200
Samples NVML GPU utilisation / memory / NVDEC utilisation every 100 ms during each run.
Writes results/bench/bench.json and a markdown table.
"""
import argparse
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_CLIPS = [  # clips with >= 1350 frames (others too short for the benchmark length)
    "clips/MD_355_and_MD_124_Gaithersburg_MD_2022_07_30_11_07_03_1.webm",
    "clips/Soundscapes_of_Norrk_ping__1____Skvallertorget.webm",
    "clips/Soundscapes_of_Norrk_ping__4____Train_station.webm",
    "clips/Starling_bathing_one_sparrow_eating_two_doves_walking_around.webm",
    "clips/Traffic_on_Slovenska_street.webm",
]


def worker(wid, gpu, paths, cfg_kw, q):
    try:
        _worker(wid, gpu, paths, cfg_kw, q)
    except BaseException as e:  # noqa: BLE001
        import traceback
        q.put(("error", wid, traceback.format_exc()))
        raise


def _worker(wid, gpu, paths, cfg_kw, q):
    import torch
    from yolo_kalman.pipeline import Pipeline, PipelineConfig
    from yolo_kalman.tracker import TrackerConfig

    torch.cuda.set_device(gpu)
    tk = cfg_kw.pop("tracker", {})
    cfg = PipelineConfig(**cfg_kw, tracker=TrackerConfig(**tk))
    pipe = Pipeline([str(ROOT / p) for p in paths], gpu=gpu, cfg=cfg)
    # warm-up (TRT context, cuDNN autotune, lazy init) on a throwaway pipeline-sized batch
    with torch.no_grad():
        pipe.det(pipe.inbuf)
        if pipe.enc is not None:
            f = torch.zeros(2, 3, 480, 854, dtype=torch.uint8, device=pipe.dev)
            pipe.enc(f, torch.tensor([[0, 0, 50, 100.]] * 64, device=pipe.dev), torch.zeros(64, device=pipe.dev))
    torch.cuda.synchronize()
    q.put(("ready", wid))
    res = pipe.run()
    q.put(("done", wid, {"gpu": gpu, "frames": pipe.frames_done, "wall": pipe.wall, "timing": pipe.timing,
                         "tracker": pipe.tracker.timing, "streams": len(paths),
                         "ids": [len({int(i) for ids, *_ in r.tracks for i in ids}) for r in res]}))


class NvmlSampler(threading.Thread):
    def __init__(self, gpus, period=0.1):
        super().__init__(daemon=True)
        import pynvml

        pynvml.nvmlInit()
        self.nv = pynvml
        self.h = [pynvml.nvmlDeviceGetHandleByIndex(g) for g in gpus]
        self.period = period
        self.samples = []
        self.stop_ev = threading.Event()

    def run(self):
        nv = self.nv
        while not self.stop_ev.is_set():
            row = []
            for h in self.h:
                u = nv.nvmlDeviceGetUtilizationRates(h)
                try:
                    dec = nv.nvmlDeviceGetDecoderUtilization(h)[0]
                except nv.NVMLError:
                    dec = -1
                m = nv.nvmlDeviceGetMemoryInfo(h)
                row.append((u.gpu, dec, m.used / 2 ** 30, nv.nvmlDeviceGetPowerUsage(h) / 1000))
            self.samples.append(row)
            time.sleep(self.period)

    def summary(self):
        out = []
        for g in range(len(self.h)):
            s = [r[g] for r in self.samples]
            if not s:
                out.append({})
                continue
            out.append({"util_mean": sum(x[0] for x in s) / len(s), "util_max": max(x[0] for x in s),
                        "nvdec_mean": sum(x[1] for x in s) / len(s), "mem_max_gb": max(x[2] for x in s),
                        "power_mean_w": sum(x[3] for x in s) / len(s)})
        return out


def _get(q, procs):
    import queue as _q
    while True:
        try:
            return q.get(timeout=5)
        except _q.Empty:
            dead = [(i, p.exitcode) for i, p in enumerate(procs) if p.exitcode not in (None, 0)]
            if dead:
                raise RuntimeError(f"worker process died: {dead}")


def run_config(n_streams, gpus, cfg_kw, wpg=1):
    paths = [BENCH_CLIPS[i % len(BENCH_CLIPS)] for i in range(n_streams)]
    use = gpus[:max(1, min(len(gpus), n_streams))]
    workers = [g for _ in range(wpg) for g in use][:n_streams]   # round-robin over GPUs
    split = [paths[k::len(workers)] for k in range(len(workers))]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(k, g, split[k], json.loads(json.dumps(cfg_kw)), q))
             for k, g in enumerate(workers)]
    for p in procs:
        p.start()
    ready = 0
    while ready < len(procs):
        m = _get(q, procs)
        if m[0] == "error":
            raise RuntimeError(f"worker {m[1]} failed:\n{m[2]}")
        ready += m[0] == "ready"
    # workers start immediately after "ready"; sample until all done
    sampler = NvmlSampler(use)
    sampler.start()
    t0 = time.time()
    results = {}
    while len(results) < len(procs):
        m = _get(q, procs)
        if m[0] == "error":
            raise RuntimeError(f"worker {m[1]} failed:\n{m[2]}")
        if m[0] == "done":
            results[m[1]] = m[2]
    wall = time.time() - t0
    sampler.stop_ev.set()
    for p in procs:
        p.join()
    frames = sum(r["frames"] for r in results.values())
    max_wall = max(r["wall"] for r in results.values())
    return {"streams": n_streams, "gpus": use, "workers": len(workers), "frames": frames, "wall_s": max_wall,
            "fps_total": frames / max_wall, "fps_per_stream": frames / max_wall / n_streams,
            "per_gpu": results, "nvml": sampler.summary(), "outer_wall_s": wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--engine", default="models/yolo26m_384x640_b64_e2e_fp16.engine")
    ap.add_argument("--no-reid", action="store_true")
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--wpg", type=int, default=1, help="worker processes per GPU")
    a = ap.parse_args()
    cfg_kw = {"engine": str(ROOT / a.engine), "chunk": a.chunk, "max_frames": a.frames,
              "use_reid": not a.no_reid, "overlap": not a.no_overlap, "profile": a.profile,
              "classes": [0, 1, 2, 3, 5, 7, 14, 15, 16]}
    out = ROOT / "results/bench"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in a.streams:
        r = run_config(n, a.gpus, dict(cfg_kw), a.wpg)
        r["tag"] = a.tag
        r["cfg"] = cfg_kw
        rows.append(r)
        t = {k: sum(v["timing"][k] for v in r["per_gpu"].values()) / len(r["per_gpu"])
             for k in next(iter(r["per_gpu"].values()))["timing"]}
        util = " ".join(f"gpu{g}:{s.get('util_mean', 0):.0f}%/dec{s.get('nvdec_mean', 0):.0f}%"
                        for g, s in zip(r["gpus"], r["nvml"]))
        print(f"[{a.tag}] streams={n:2d} gpus={r['gpus']} workers={r['workers']} fps_total={r['fps_total']:.0f} "
              f"per_stream={r['fps_per_stream']:.0f} wall={r['wall_s']:.1f}s  {util}  "
              f"split(s/gpu)=" + " ".join(f"{k}:{v:.2f}" for k, v in t.items()), flush=True)
    f = out / f"bench_{a.tag}.json"
    f.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
