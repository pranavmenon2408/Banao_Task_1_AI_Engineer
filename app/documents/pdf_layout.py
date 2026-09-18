"""Column-aware PDF text extraction with PyMuPDF.

Many resume builders draw multi-column layouts (sidebar + main column) row by row across the page, so reading text
in content-stream order interleaves the columns line by line. PyMuPDF exposes every text span with its bounding box,
which lets us choose the reading order ourselves:

1. Split each line into segments wherever two spans are far apart (text on one baseline in two columns is
   otherwise reported as a single line).
2. Find a vertical gutter: an x-range that no body segment crosses, with substantial text on both sides
   (>= 8% of the page's characters and >= 5 lines each) and few lines spanning it. These thresholds keep a
   right-aligned column of dates attached to its roles instead of being read as a second column.
3. Lines that cross the gutter (a name banner, a full-width heading) split the page into bands; within each band
   the left column is read top to bottom, then the right column.

Pages without a qualifying gutter are read top to bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

MIN_GUTTER_PT = 12
SIDE_MIN_SHARE = 0.08  # a sidebar is typically 10-25% of the text; a right-aligned dates column is ~3-5%
SIDE_MIN_LINES = 5
MAX_CROSSING_SHARE = 0.15  # at most this share of lines may span the gutter (headers, name banner)


@dataclass
class Line:
    """A run of text on one baseline with its bounding box (PDF points)."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def page_lines(page: pymupdf.Page) -> list[Line]:
    """Text segments with boxes. PyMuPDF merges text that shares a baseline into one line even when it
    sits in two different columns, so a line is split wherever two spans are far apart horizontally."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for ln in block["lines"]:
            seg: list[dict] = []
            for span in ln["spans"]:
                if seg and span["bbox"][0] - seg[-1]["bbox"][2] > max(MIN_GUTTER_PT, 2 * span["size"]):
                    out.append(_segment(seg))
                    seg = []
                seg.append(span)
            if seg:
                out.append(_segment(seg))
    return [ln for ln in out if ln.text]


def _segment(spans: list[dict]) -> Line:
    x0 = min(s["bbox"][0] for s in spans)
    y0 = min(s["bbox"][1] for s in spans)
    x1 = max(s["bbox"][2] for s in spans)
    y1 = max(s["bbox"][3] for s in spans)
    return Line(x0, y0, x1, y1, "".join(s["text"] for s in spans).strip())


def find_gutter(lines: list[Line], width: float) -> float | None:
    """Return the x-coordinate of a column gutter, or None for a single-column page."""
    body = [ln for ln in lines if (ln.x1 - ln.x0) < 0.55 * width]
    if len(body) < 2 * SIDE_MIN_LINES:
        return None
    spans = sorted((ln.x0, ln.x1) for ln in body)
    merged = [list(spans[0])]
    for x0, x1 in spans[1:]:
        if x0 <= merged[-1][1] + 2:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]
    gaps = [g for g in gaps if g[1] - g[0] >= MIN_GUTTER_PT and 0.2 * width <= (g[0] + g[1]) / 2 <= 0.8 * width]
    total = sum(len(ln.text) for ln in lines)  # share of ALL page text, not just narrow lines
    for g0, g1 in sorted(gaps, key=lambda g: g[1] - g[0], reverse=True):
        mid = (g0 + g1) / 2
        left = [ln for ln in body if ln.x1 <= g0 + 1]
        right = [ln for ln in body if ln.x0 >= g1 - 1]
        crossing = sum(1 for ln in lines if ln.x0 < mid < ln.x1)
        lc, rc = sum(len(ln.text) for ln in left), sum(len(ln.text) for ln in right)
        # In a genuine two-column page only a header or two spans the gutter. If many lines cross it
        # (e.g. full-width bullets with right-aligned dates on the role lines) the page is one column.
        if (
            min(len(left), len(right)) >= SIDE_MIN_LINES
            and lc >= SIDE_MIN_SHARE * total
            and rc >= SIDE_MIN_SHARE * total
            and crossing <= MAX_CROSSING_SHARE * len(lines)
        ):
            return mid
    return None


def _join(lines: list[Line]) -> list[str]:
    """Top-to-bottom; lines on the same row are joined, and a larger vertical gap becomes a blank line."""
    lines = sorted(lines, key=lambda ln: (round(ln.y0, 0), ln.x0))
    rows: list[list[Line]] = []
    for ln in lines:
        if rows and abs(ln.y0 - rows[-1][0].y0) < 3:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    out: list[str] = []
    prev_bottom, heights = None, [r[0].y1 - r[0].y0 for r in rows] or [10]
    typical = sorted(heights)[len(heights) // 2]
    for r in rows:
        r.sort(key=lambda ln: ln.x0)
        if prev_bottom is not None and r[0].y0 - prev_bottom > 1.2 * typical:
            out.append("")
        out.append("  ".join(ln.text for ln in r))
        prev_bottom = max(ln.y1 for ln in r)
    return out


def order_page(lines: list[Line], width: float) -> tuple[list[str], bool]:
    """Order a page's lines for reading; returns (lines, whether columns were detected)."""
    gutter = find_gutter(lines, width)
    if gutter is None:
        return _join(lines), False
    out: list[str] = []
    left: list[Line] = []
    right: list[Line] = []

    def flush() -> None:
        nonlocal left, right
        out.extend(_join(left))
        if left and right:
            out.append("")
        out.extend(_join(right))
        left, right = [], []

    for ln in sorted(lines, key=lambda ln: ln.y0):
        if ln.x0 < gutter < ln.x1:  # full-width line: ends the current band
            flush()
            out.append(ln.text)
        elif ln.x1 <= gutter:
            left.append(ln)
        else:
            right.append(ln)
    flush()
    return out, True


def extract(doc: pymupdf.Document) -> tuple[str, bool]:
    """Return (text, whether any page was detected as multi-column)."""
    pages, multi = [], False
    for page in doc:
        lines, cols = order_page(page_lines(page), page.rect.width)
        multi |= cols
        pages.append("\n".join(lines))
    return "\n\n".join(pages), multi
