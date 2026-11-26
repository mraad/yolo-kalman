"""Download 480p VP9 transcodes of the Wikimedia Commons clips listed in meta/urls.tsv.

Downloads are untrusted data: they go into clips/ and are only ever read as media.
Uses a generic User-Agent.  Falls back to 360p when no 480p transcode exists.
"""
import csv
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = "yolo-kalman-clipfetch/0.1 (research tracker test; generic client) python-urllib"


def transcode_url(orig: str, res: str) -> str:
    p = urllib.parse.urlparse(orig).path  # /wikipedia/commons/0/0b/Calle_preciados.ogv
    parts = p.split("/")
    a, ab, fname = parts[-3], parts[-2], parts[-1]
    return f"https://upload.wikimedia.org/wikipedia/commons/transcoded/{a}/{ab}/{fname}/{fname}.{res}.vp9.webm"


def safe_name(title: str) -> str:
    stem = title.rsplit(".", 1)[0]
    return "".join(c if (c.isascii() and c.isalnum()) else "_" for c in stem).strip("_")[:60] + ".webm"


def fetch(url: str, dst: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        return True
    except Exception as e:  # noqa: BLE001
        print("  fail", url, e, file=sys.stderr)
        dst.unlink(missing_ok=True)
        return False


def main():
    out = ROOT / "clips"
    out.mkdir(exist_ok=True)
    rows = list(csv.reader(open(ROOT / "meta/urls.tsv"), delimiter="\t"))
    index = []
    for title, url, page, lic, dur, h in rows:
        dst = out / safe_name(title)
        got = dst.exists() and dst.stat().st_size > 0
        res_used = "cached"
        if not got:
            for res in ("480p", "360p"):
                if fetch(transcode_url(url, res), dst):
                    got, res_used = True, res
                    break
                time.sleep(6)
        print(("ok  " if got else "MISS"), res_used, dst.name, flush=True)
        if got:
            index.append((dst.name, title, page, lic, dur, h))
        time.sleep(6)
    with open(ROOT / "meta/clips_index.tsv", "w") as f:
        f.write("file\ttitle\tpage\tlicence\tduration_s\torig_height\n")
        for r in index:
            f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    main()
