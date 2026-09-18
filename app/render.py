"""Render the criteria and profile as prompt text for the scorer.

JSON is the right format for what the model *returns* (we validate it with Pydantic), but not
necessarily for what it *reads*: it costs extra tokens on quotes/keys/indentation, escapes characters
the scorer must copy verbatim, and makes criteria and resume evidence look alike (the scorer once
quoted a criterion as resume evidence; see docs/DEVLOG.md). The Markdown rendering keeps the profile's
structure (bullets under their role) with the text exactly as extracted. The format is switchable
in config/scoring.yaml so both can be compared with scripts/calibrate.py.
"""
from __future__ import annotations

import json

from app.schemas import Criterion, ResumeProfile

IMPORTANCE_LABEL = {"must_have": "MUST HAVE", "important": "IMPORTANT", "nice_to_have": "NICE TO HAVE"}


def criteria_text(criteria: list[Criterion], fmt: str) -> str:
    if fmt == "json":
        return json.dumps([c.model_dump(exclude={"jd_evidence"}) for c in criteria], indent=1, ensure_ascii=False)
    return "\n".join(f"[{c.id}] {c.name} ({IMPORTANCE_LABEL[c.importance]}, {c.category})\n    Requirement: {c.description}"
                     for c in criteria)


def profile_text(p: ResumeProfile, fmt: str) -> str:
    # The summary is written by Agent 1, not copied from the resume, so it is not evidence; hiding it
    # from the scorer stops it quoting generated text (observed in the first run, see docs/DEVLOG.md).
    if fmt == "json":
        return p.model_dump_json(indent=1, exclude_none=True, exclude={"summary"})
    out: list[str] = []
    if p.candidate_name or p.headline:
        out.append(" | ".join(x for x in (p.candidate_name, p.headline) if x))
    if p.total_experience_years is not None:
        out.append(f"Total professional experience (computed by extractor): {p.total_experience_years} years")
    if p.experience:
        out.append("\n## EXPERIENCE")
        for e in p.experience:
            dates = " - ".join(x for x in (e.start, e.end) if x)
            dur = f", {e.duration_months} months" if e.duration_months else ""
            out.append(f"### {e.title} | {e.company} ({dates}{dur})")
            out += [f"- {h}" for h in e.highlights]
    if p.projects:
        out.append("\n## PROJECTS")
        for pr in p.projects:
            tech = f" [{', '.join(pr.technologies)}]" if pr.technologies else ""
            out.append(f"- {pr.name}: {pr.description}{tech}")
    if p.skills:
        out.append("\n## SKILLS")
        out.append(", ".join(p.skills))
    if p.education:
        out.append("\n## EDUCATION")
        for ed in p.education:
            out.append("- " + ", ".join(x for x in (ed.degree, ed.field, ed.institution, ed.year) if x))
    if p.certifications:
        out.append("\n## CERTIFICATIONS")
        out += [f"- {c}" for c in p.certifications]
    if p.other:
        out.append("\n## OTHER")
        out += [f"- {o}" for o in p.other]
    return "\n".join(out).strip()
