"""Verify that quoted evidence actually appears in the source document.

The scorer is told to quote the resume verbatim. This module checks that it did, so a hallucinated
"Led a team of 12 engineers" can't raise a score. Exact normalised substring match first; if that
fails, a fuzzy match against same-length word windows, which tolerates the small differences
PDF extraction introduces (hyphenation, bullets, ligatures).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.schemas import EvidenceCheck


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"[•●▪◦·*]", " ", s)
    s = re.sub(r"[^\w%+#./\-' ]+", " ", s)
    return " ".join(s.split())


class Grounder:
    def __init__(self, source_text: str):
        self.norm = _norm(source_text)
        self.words = self.norm.split()

    def similarity(self, quote: str) -> float:
        best = self._similarity(quote)
        if best >= 0.97:
            return best
        # The scorer often quotes a line the renderer assembled from several extracted fields
        # ("B.E. Computer Science, RV College of Engineering, 2018"; "Python, FastAPI, Kafka"). Such a
        # line is grounded if every fragment is. Observed: a correct education level was cut because
        # the joined line did not exist verbatim (docs/DEVLOG.md).
        frags = [f.strip() for f in re.split(r"\s*[,|;]\s*", quote) if len(f.strip()) >= 2]
        if len(frags) > 1:
            best = max(best, min(self._similarity(f) for f in frags))
        return best

    def _similarity(self, quote: str) -> float:
        q = _norm(quote.strip(" .\"'…").replace("...", " "))
        if not q:
            return 0.0
        if q in self.norm:
            return 1.0
        qw = q.split()
        n = len(qw)
        if n == 0 or not self.words:
            return 0.0
        best = 0.0
        first = set(qw[:3])
        for i in range(0, max(1, len(self.words) - n + 1)):
            # cheap pre-filter: a real match shares at least one of the first three words nearby
            if not first.intersection(self.words[i:i + 3]):
                continue
            window = " ".join(self.words[i:i + n])
            sm = SequenceMatcher(None, q, window, autojunk=False)
            if sm.quick_ratio() <= best:
                continue
            best = max(best, sm.ratio())
            if best > 0.97:
                break
        return round(best, 3)

    def check(self, quotes: list[str], threshold: float) -> list[EvidenceCheck]:
        out = []
        for q in quotes:
            sim = self.similarity(q)
            out.append(EvidenceCheck(quote=q, found=sim >= threshold, similarity=sim))
        return out
