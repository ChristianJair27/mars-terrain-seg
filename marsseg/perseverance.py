"""Fetch a few recent Perseverance NAVCAM frames from NASA's public raw-image feed (mars.nasa.gov).

Used only for a qualitative out-of-domain look: different rover, camera (colour, 20 MP NAVCAM vs.
Curiosity's 1 MP greyscale), site (Jezero crater vs. Gale crater). Raw images are unprocessed engineering
products; credit NASA/JPL-Caltech."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://mars.nasa.gov/rss/api/"
UA = {"User-Agent": "Mozilla/5.0 (marsseg research script)"}


def query(n: int = 100, page: int = 0, camera: str = "NAVCAM_LEFT") -> list[dict]:
    q = {"feed": "raw_images", "category": "mars2020", "feedtype": "json", "num": n, "page": page,
         "order": "sol desc", "search": f"|{camera}"}
    url = API + "?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)["images"]


def pick(images: list[dict], k: int = 6, el_range=(-40.0, -8.0)) -> list[dict]:
    """Full-frame (not thumbnail/subframe) images looking down at the terrain near the rover,
    at most one per sol so the set spans several sites."""
    out, sols = [], set()
    for im in images:
        ext = im.get("extended", {})
        try:
            el = float(ext.get("mastEl", "nan"))
        except ValueError:
            continue
        if im.get("sample_type") != "Full" or not (el_range[0] <= el <= el_range[1]):
            continue
        if ext.get("subframeRect") != "(1,1,5120,3840)":  # whole sensor, not a strip
            continue
        if im["sol"] in sols:
            continue
        sols.add(im["sol"])
        out.append(im)
        if len(out) == k:
            break
    return out


def fetch(out: Path, k: int = 6, pages: int = 20) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    images = []
    for p in range(pages):
        images += query(page=p)
    chosen = pick(images, k)
    paths, meta = [], []
    for im in chosen:
        url = im["image_files"].get("large") or im["image_files"]["full_res"]
        dst = out / (im["imageid"] + Path(urllib.parse.urlparse(url).path).suffix.lower())
        if not dst.exists():
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                dst.write_bytes(r.read())
        paths.append(dst)
        meta.append({"file": dst.name, "sol": im["sol"], "imageid": im["imageid"], "url": url,
                     "date_taken_utc": im.get("date_taken_utc"), "mastEl": im["extended"].get("mastEl"),
                     "credit": "NASA/JPL-Caltech"})
    (out / "perseverance_meta.json").write_text(json.dumps(meta, indent=1))
    return paths
