"""On-disk JSON cache for extracted criteria and resume profiles.

Caching matters for consistency as much as for speed. A job description's criteria are extracted once and reused
for every resume scored against it, so all candidates for a role are judged against the same rubric. Keys include
the model and prompt version, so a prompt or model change never reuses stale results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.core.config import ROOT, get_settings


def text_hash(*parts: str) -> str:
    """Stable 16-hex-char key for the given strings, insensitive to case and whitespace differences."""
    h = hashlib.sha256()
    for p in parts:
        h.update(" ".join(p.split()).lower().encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class JsonCache:
    """A directory of JSON files, one per key. Writes are atomic (write to a temp file, then rename)."""

    def __init__(self, namespace: str, base: Path | None = None) -> None:
        base = base or Path(get_settings().data_dir)
        if not base.is_absolute():
            base = ROOT / base
        self.dir = base / "cache" / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached value, or None if missing or unreadable."""
        p = self.dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store a value under `key`, replacing any previous one."""
        tmp = self.dir / f"{key}.json.tmp"
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.dir / f"{key}.json")
