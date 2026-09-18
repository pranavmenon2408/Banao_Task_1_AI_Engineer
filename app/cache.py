"""Tiny on-disk JSON cache (local storage only, no database).

Its main job is calibration, not speed: the job description's criteria are extracted once and reused
for every resume scored against that JD. If criteria were re-extracted per request, two resumes could
be scored against slightly different rubrics (7 criteria vs 9, "important" vs "must_have"), which on
its own moves overall scores by 10+ points.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.config import ROOT, get_settings


def text_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(" ".join(p.split()).lower().encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class JsonCache:
    def __init__(self, namespace: str, base: Path | None = None):
        base = base or Path(get_settings().data_dir)
        if not base.is_absolute():
            base = ROOT / base
        self.dir = base / "cache" / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        p = self.dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: dict) -> None:
        tmp = self.dir / f"{key}.json.tmp"
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.dir / f"{key}.json")
