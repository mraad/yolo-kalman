"""Collect results/bench/bench_<tag>*.json into a markdown table (results/bench/table.md)."""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pat = sys.argv[1] if len(sys.argv) > 1 else "twogpu_s*"
rows = []
for f in sorted(glob.glob(str(ROOT / f"results/bench/bench_{pat}.json"))):
    rows += json.load(open(f))
rows.sort(key=lambda r: r["streams"])
L = ["| solver | streams | GPUs | frames | wall s | FPS total | FPS/stream | GPU util % (mean per GPU) | NVDEC % | decode s | letterbox s | detect s | embed s | track s | tracker pre / match / post s |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    pg = list(r["per_gpu"].values())
    t = {k: sum(v["timing"][k] for v in pg) / len(pg) for k in pg[0]["timing"]}
    tr = {k: sum(v["tracker"][k] for v in pg) / len(pg) for k in pg[0]["tracker"]}
    match = tr.get("match", tr.get("cpu_match", 0.0))
    solver = r["cfg"].get("tracker", {}).get("assign", "lapjv")
    util = " / ".join(f"{s.get('util_mean', 0):.0f}" for s in r["nvml"])
    dec = " / ".join(f"{s.get('nvdec_mean', 0):.0f}" for s in r["nvml"])
    L.append(f"| {solver} | {r['streams']} | {len(r['gpus'])} | {r['frames']} | {r['wall_s']:.1f} | {r['fps_total']:.0f} | "
             f"{r['fps_per_stream']:.0f} | {util} | {dec} | {t['decode']:.2f} | {t['preprocess']:.2f} | {t['detect']:.2f} | "
             f"{t['embed']:.2f} | {t['track']:.2f} | {tr['gpu_pre']:.2f} / {match:.2f} / {tr['gpu_post']:.2f} |")
out = "\n".join(L) + "\n"
(ROOT / f"results/bench/table_{pat.replace('*', '')}.md").write_text(out)
print(out)
