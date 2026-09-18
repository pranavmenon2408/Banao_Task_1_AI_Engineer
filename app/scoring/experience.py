"""Years of experience computed from extracted role dates.

Whether "3.2 years" meets "5+ years" is arithmetic, so it is computed here rather than judged by the model. Each
role's start and end ("Mar 2022", "2019", "06/2020", "Present") is parsed, overlapping periods are merged so
concurrent roles are not double counted, and internships are excluded by default.
"""

from __future__ import annotations

import re
from datetime import date

from app.core.schemas import ExperienceItem

MONTHS = {
    m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)
}
PRESENT = {"present", "current", "now", "today", "ongoing", "till date", "date"}
INTERN = re.compile(r"\bintern(ship)?\b|\btrainee\b", re.I)


def parse_month(s: str | None, *, end: bool = False, today: date | None = None) -> int | None:
    """Return months since year 0 (year*12 + month-1), or None if unparseable."""
    if not s:
        return None
    t = s.strip().lower()
    today = today or date.today()
    if any(p in t for p in PRESENT):
        return today.year * 12 + today.month - 1
    y = re.search(r"(19|20)\d{2}", t)
    if not y:
        return None
    year = int(y.group(0))
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", t)
    if m:
        month = MONTHS[m.group(1)]
    else:
        num = re.match(r"\s*(\d{1,2})\s*[/.-]\s*(19|20)\d{2}", t)
        month = int(num.group(1)) if num and 1 <= int(num.group(1)) <= 12 else (12 if end else 1)
    return year * 12 + month - 1


def role_months(e: ExperienceItem, today: date | None = None) -> tuple[int, int] | None:
    """A role's (start, end) in months, end exclusive; None if the dates cannot be parsed."""
    s, en = parse_month(e.start, today=today), parse_month(e.end, end=True, today=today)
    if s is None:
        return None
    if en is None:
        if e.duration_months:
            return s, s + e.duration_months
        return None
    return (s, en + 1) if en >= s else None


def total_years(
    experience: list[ExperienceItem], include_internships: bool = False, today: date | None = None
) -> float | None:
    """Total professional years across roles, with overlaps merged; None if no dates parse."""
    spans = []
    for e in experience:
        if not include_internships and INTERN.search(e.title or ""):
            continue
        span = role_months(e, today)
        if span:
            spans.append(span)
    if not spans:
        return None
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return round(sum(e - s for s, e in merged) / 12, 1)
