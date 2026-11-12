"""Fetch the Curiosity (MSL) NAVCAM part of AI4Mars from the official release.

Source: AI4Mars v0.6, NASA JPL, Zenodo record 15995036 (linked from data.nasa.gov),
file ai4mars-dataset-merged-0.6.zip (16.2 GB, all four rovers). We read the ZIP central directory
with HTTP range requests and download only the members under the selected prefixes, grouping
neighbouring members into large contiguous spans so that ~50k small files cost a few hundred requests.
"""
from __future__ import annotations

import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .remotezip import HTTPRangeFile

URL = "https://zenodo.org/records/15995036/files/ai4mars-dataset-merged-0.6.zip?download=1"
ROOT = "ai4mars-dataset-merged-0.6/"

# Curiosity NAVCAM: images, rover masks, 30 m range masks, crowd train labels, expert test labels,
# plus the top-level documentation files.
MSL_PREFIXES = [
    "msl/ncam/images/edr/",
    "msl/ncam/images/mxy/",
    "msl/ncam/images/rng-30m/",
    "msl/ncam/labels/",
]
# Perseverance NAVCAM with geologist labels, for an out-of-domain check (small).
M2020_PREFIXES = ["m2020/images/ncam/", "m2020/labels/M2020_GEO/ncam/"]


def _member_data(span: bytes, span_start: int, info: zipfile.ZipInfo) -> bytes:
    off = info.header_offset - span_start
    sig, *_rest = struct.unpack("<IHHHHHIIIHH", span[off:off + 30])
    if sig != 0x04034B50:
        raise IOError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", span[off + 26:off + 30])
    start = off + 30 + name_len + extra_len
    raw = span[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        return raw
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(raw, -15)
    raise IOError(f"unsupported compression {info.compress_type} for {info.filename}")


def plan(prefixes: list[str], url: str = URL):
    rf = HTTPRangeFile(url)
    zf = zipfile.ZipFile(rf)
    infos = sorted(zf.infolist(), key=lambda i: i.header_offset)
    ends = {i.filename: (infos[k + 1].header_offset if k + 1 < len(infos) else rf.size)
            for k, i in enumerate(infos)}
    want = [i for i in infos if not i.is_dir()
            and (i.filename.count("/") == 1 or any(i.filename.startswith(ROOT + p) for p in prefixes))]
    return rf, want, ends


def spans(want, ends, max_span: int = 64 << 20, max_gap: int = 1 << 20):
    """Group members into contiguous byte ranges no larger than max_span."""
    groups, cur = [], []
    for info in want:
        if cur:
            gap = info.header_offset - ends[cur[-1].filename]
            size = ends[info.filename] - cur[0].header_offset
            if gap > max_gap or size > max_span:
                groups.append(cur)
                cur = []
        cur.append(info)
    if cur:
        groups.append(cur)
    return groups


def fetch(out: Path, prefixes: list[str] = MSL_PREFIXES, workers: int = 6, url: str = URL) -> int:
    out = Path(out)
    rf, want, ends = plan(prefixes, url)
    todo = [i for i in want if not (out / i.filename[len(ROOT):]).exists()]
    groups = spans(todo, ends)
    total = sum(ends[g[-1].filename] - g[0].header_offset for g in groups)
    print(f"{len(want)} members selected, {len(todo)} to fetch in {len(groups)} spans, {total / 1e9:.2f} GB")

    def work(g):
        start, end = g[0].header_offset, ends[g[-1].filename]
        data = HTTPRangeFile.__new__(HTTPRangeFile)  # reuse the retrying getter without a HEAD request
        data.url, data.retries = url, 5
        span = data._get(start, end - 1)[0]
        for info in g:
            dst = out / info.filename[len(ROOT):]
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(_member_data(span, start, info))
            tmp.replace(dst)
        return len(g), end - start

    done_n = done_b = 0
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(work, g) for g in groups]):
            n, b = fut.result()
            done_n += n
            done_b += b
            print(f"  {done_n}/{len(todo)} files, {done_b / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
    return done_n
