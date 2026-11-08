"""Download helpers for NEMWeb (https://nemweb.com.au)."""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests

BASE = "https://nemweb.com.au"
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "nemcast/0.1 (personal research)"


def list_files(folder: str) -> list[str]:
    """List zip file names in a NEMWeb folder such as `Reports/Archive/P5_Reports`."""
    html = _get(f"{BASE}/{folder.strip('/')}/").text
    return sorted(set(re.findall(r"(PUBLIC_[A-Za-z0-9_]+\.zip)", html, flags=re.IGNORECASE)))


def download(folder: str, name: str, dest_dir: str | Path) -> Path:
    """Download `folder/name` into `dest_dir` unless already present."""
    dest = Path(dest_dir) / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with _get(f"{BASE}/{folder.strip('/')}/{name}", stream=True) as r, open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def _get(url: str, stream: bool = False, retries: int = 4) -> requests.Response:
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, timeout=120, stream=stream)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")
