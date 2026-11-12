"""Read members of a remote ZIP archive with HTTP range requests.

The official AI4Mars release (Zenodo record 15995036) is a single 16.2 GB ZIP holding all four rovers.
We only need Curiosity (MSL) NAVCAM images and labels, so we read the archive's central directory
through range requests and pull only the members we want. Nothing in the archive is executed.
"""
from __future__ import annotations

import io
import time
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130 Safari/537.36")


class HTTPRangeFile(io.RawIOBase):
    """A seekable, read-only file object backed by HTTP range requests, with a small block cache."""

    def __init__(self, url: str, block: int = 1 << 20, retries: int = 5):
        self.url, self.block, self.retries = url, block, retries
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        head = self._get(0, 0)
        self.size = int(head[1].split("/")[-1])

    def _get(self, start: int, end: int) -> tuple[bytes, str]:
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise IOError(f"server ignored the range request (HTTP {r.status})")
                    return r.read(), r.headers["Content-Range"]
            except Exception:  # noqa: BLE001 - network hiccups: retry with backoff
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            start = i * self.block
            self.cache[i] = self._get(start, min(start + self.block, self.size) - 1)[0]
        return self.cache[i]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n > 4 * self.block:  # large member: one direct request, no caching
            data = self._get(self.pos, self.pos + n - 1)[0]
            self.pos += len(data)
            return data
        out = bytearray()
        while n > 0:
            i, off = divmod(self.pos, self.block)
            chunk = self._block(i)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
