"""Build docs/index.html (usage: python3 docs/build_docs.py) from docs/index.src.html: embed images, throughput chart and table rows."""
import base64, json, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
img = root / "docs/img"
src = (root / "docs/index.src.html").read_text()
a = json.loads((root / "results/bench/bench_audit_gpu_hungarian.json").read_text())
b = json.loads((root / "results/bench/bench_audit_lapjv.json").read_text())
assert [r["streams"] for r in a] == [r["streams"] for r in b]

# chart: grouped bars, y 0..6000
W, H, L, R, T, B = 640, 300, 60, 14, 24, 44
pw, ph, ymax = W - L - R, H - T - B, 6000
y = lambda v: T + ph - v / ymax * ph
g = pw / len(a); bw = 30
out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Grouped bars of total frames per second for 1, 4, 8, 16 and 32 streams; GPU Hungarian is higher at every stream count.">']
for t in range(0, ymax + 1, 1500):
    out.append(f'<line x1="{L}" x2="{W-R}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/>')
    out.append(f'<text x="{L-8}" y="{y(t)+4:.1f}" text-anchor="end" class="t-sm">{t:,}</text>')
out.append(f'<text x="{L-8}" y="{T-10}" text-anchor="end" class="t-sm">fps</text>')
for i, (ra, rb) in enumerate(zip(a, b)):
    cx = L + g * (i + 0.5)
    for k, (r, cls, dx) in enumerate(((ra, "bar1", -bw - 1), (rb, "bar2", 1))):
        v = r["fps_total"]; top = y(v); h = T + ph - top
        # 4px rounded top, square at baseline
        x0 = cx + dx
        out.append(f'<path class="{cls}" d="M{x0:.1f},{T+ph} V{top+4:.1f} Q{x0:.1f},{top:.1f} {x0+4:.1f},{top:.1f} H{x0+bw-4:.1f} '
                   f'Q{x0+bw:.1f},{top:.1f} {x0+bw:.1f},{top+4:.1f} V{T+ph} Z"><title>{r["streams"]} streams, '
                   f'{"GPU Hungarian" if k == 0 else "lapjv"}: {v:,.0f} fps</title></path>')
        out.append(f'<text x="{x0+bw/2:.1f}" y="{top-6:.1f}" text-anchor="middle" class="t-sm">{v:,.0f}</text>')
    out.append(f'<text x="{cx:.1f}" y="{T+ph+18}" text-anchor="middle">{ra["streams"]}</text>')
out.append(f'<line x1="{L}" x2="{W-R}" y1="{T+ph}" y2="{T+ph}" class="ln"/>')
out.append(f'<text x="{L+pw/2:.1f}" y="{H-6}" text-anchor="middle" class="t-sm">parallel streams (2 GPUs; 1 GPU for 1 stream)</text>')
out.append("</svg>")
chart = "\n".join(out)

rows = []
for ra, rb in zip(a, b):
    util = " / ".join(f'{s.get("util_mean", 0):.0f} %' for s in ra["nvml"])
    pg = list(ra["per_gpu"].values()); n = len(pg)
    tr = [sum(v["tracker"][k] for v in pg) / n for k in ("gpu_pre", "match", "gpu_post")]
    rows.append(f'<tr><td class="n">{ra["streams"]}</td><td class="n">{ra["fps_total"]:,.0f}</td><td class="n">{rb["fps_total"]:,.0f}</td>'
                f'<td class="n">{ra["fps_total"]/rb["fps_total"]:.2f}×</td><td class="n">{util}</td>'
                f'<td class="n">{tr[0]:.2f} / {tr[1]:.2f} / {tr[2]:.2f}</td></tr>')
src = src.replace("{{CHART}}", chart).replace("{{TPUT_ROWS}}", "\n".join(rows))
for name in ("strip_g120", "traffic", "doves", "station"):
    data = base64.b64encode((img / f"{name}.jpg").read_bytes()).decode()
    src = src.replace("{{IMG:%s}}" % name, "data:image/jpeg;base64," + data)
assert "{{" not in src, src[src.index("{{"):src.index("{{") + 40]
(root / "docs/index.html").write_text(src)
print(len(src) // 1024, "KiB")
