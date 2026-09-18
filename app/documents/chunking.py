"""Section-aware chunking for the resume-extraction agent.

Resumes are short (1-3 pages is roughly 600-2,000 tokens), so in the common case the whole resume is
one chunk and no chunking happens. Chunking only kicks in for long CVs (academic CVs, 6+ page
resumes). When it does, we split on section headings rather than fixed windows: a fixed
512-token window would separate a job title from its bullets, and the extractor would then
attribute achievements to the wrong role. Sections are packed greedily into chunks up to the token
budget; only a single section that is larger than the budget gets split, on paragraph/line
boundaries, with its heading repeated so the chunk keeps its context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HEADINGS = {
    "summary", "professional summary", "profile", "objective", "about", "about me",
    "experience", "work experience", "professional experience", "employment", "employment history",
    "work history", "career history", "relevant experience",
    "education", "academic background", "qualifications",
    "skills", "technical skills", "core skills", "key skills", "core competencies", "technologies",
    "projects", "personal projects", "key projects", "selected projects",
    "certifications", "certificates", "licenses", "courses", "training",
    "publications", "awards", "achievements", "honors", "languages", "interests", "volunteering",
    "volunteer experience", "leadership", "activities", "references", "research",
}


def estimate_tokens(text: str) -> int:
    """~4 characters per token for English prose; good enough for budgeting, no tokenizer download."""
    return max(1, len(text) // 4)


def _is_heading(line: str) -> bool:
    s = line.strip().rstrip(":").strip()
    if not s or len(s) > 40:
        return False
    if s.lower() in HEADINGS:
        return True
    letters = re.sub(r"[^A-Za-z]", "", s)
    return len(letters) >= 4 and s.isupper() and len(s.split()) <= 4


@dataclass
class Section:
    heading: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.heading}\n{self.body}".strip() if self.heading else self.body.strip()


def split_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    heading, buf = "", []
    for line in text.splitlines():
        if _is_heading(line):
            if heading or "".join(buf).strip():
                sections.append(Section(heading, "\n".join(buf).strip()))
            heading, buf = line.strip(), []
        else:
            buf.append(line)
    if heading or "".join(buf).strip():
        sections.append(Section(heading, "\n".join(buf).strip()))
    return sections


def _split_oversized(section: Section, budget: int) -> list[str]:
    """Split one section that alone exceeds the budget, repeating its heading on every piece."""
    units = [u.strip() for u in re.split(r"\n\s*\n", section.body) if u.strip()]
    if len(units) == 1:
        units = [u.strip() for u in section.body.splitlines() if u.strip()]
    head = f"{section.heading} (continued)\n" if section.heading else ""
    max_chars = budget * 4
    pieces: list[str] = []
    for u in units:  # hard-cut pathological single lines that alone exceed the budget
        pieces.extend(u[i:i + max_chars] for i in range(0, len(u), max_chars))

    out: list[str] = []
    cur: list[str] = []
    for p in pieces:
        if cur and estimate_tokens(head + "\n".join(cur + [p])) > budget:
            out.append("\n".join(cur))
            cur = []
        cur.append(p)
    if cur:
        out.append("\n".join(cur))
    first = f"{section.heading}\n" if section.heading else ""
    return [(first if i == 0 else head) + body for i, body in enumerate(out)]


def chunk_resume(text: str, budget_tokens: int = 3000) -> list[str]:
    if estimate_tokens(text) <= budget_tokens:
        return [text]
    chunks: list[str] = []
    cur = ""
    for sec in split_sections(text):
        st = sec.text
        if estimate_tokens(st) > budget_tokens:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.extend(_split_oversized(sec, budget_tokens))
            continue
        if cur and estimate_tokens(cur + "\n\n" + st) > budget_tokens:
            chunks.append(cur.strip())
            cur = ""
        cur += st + "\n\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks
