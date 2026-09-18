"""Render the criteria and profile as prompt text for the scorer.

JSON is the right format for what the model *returns* (we validate it with Pydantic), but not
necessarily for what it *reads*: it costs extra tokens on quotes/keys/indentation, escapes characters
the scorer must copy verbatim, and makes criteria and resume evidence look alike (the scorer once
quoted a criterion as resume evidence; see docs/DEVLOG.md). The Markdown rendering keeps the profile's
structure (bullets under their role) with the text exactly as extracted. The format is switchable
in config/scoring.yaml so both can be compared with scripts/calibrate.py.

`parts` restricts which resume sections are shown; sectioned scoring uses it so each JD section is
judged only against the resume sections that can evidence it.
"""
from __future__ import annotations

import json

from app.scoring.experience import role_months, total_years
from app.core.schemas import Criterion, ResumeProfile

IMPORTANCE_LABEL = {"must_have": "MUST HAVE", "important": "IMPORTANT", "nice_to_have": "NICE TO HAVE"}
ALL_PARTS = frozenset({"header", "experience", "projects", "skills", "education", "certifications", "other"})


def criteria_text(criteria: list[Criterion], fmt: str) -> str:
    if fmt == "json":
        return json.dumps([c.model_dump(exclude={"jd_evidence"}) for c in criteria], indent=1, ensure_ascii=False)
    return "\n".join(f"[{c.id}] {c.name} ({IMPORTANCE_LABEL[c.importance]}, {c.category})\n    Requirement: {c.description}"
                     for c in criteria)


def profile_text(p: ResumeProfile, fmt: str, parts: frozenset[str] = ALL_PARTS, computed_years: bool = False) -> str:
    # The summary is written by Agent 1, not copied from the resume, so it is not evidence; hiding it
    # from the scorer stops it quoting generated text (observed in the first run, see docs/DEVLOG.md).
    if fmt == "json":
        keep = {"header": ["candidate_name", "headline", "total_experience_years"], "experience": ["experience"],
                "projects": ["projects"], "skills": ["skills"], "education": ["education"],
                "certifications": ["certifications"], "other": ["other"]}
        include = {f for part in parts for f in keep[part]}
        return p.model_dump_json(indent=1, exclude_none=True, include=include)
    out: list[str] = []
    if "header" in parts:
        if p.candidate_name or p.headline:
            out.append(" | ".join(x for x in (p.candidate_name, p.headline) if x))
        yrs = total_years(p.experience) if computed_years else None
        if yrs is not None:
            out.append(f"Total professional experience (computed in code from role dates, overlaps merged, "
                       f"internships excluded): {yrs} years")
        elif p.total_experience_years is not None:
            out.append(f"Total professional experience (computed by extractor): {p.total_experience_years} years")
    if "experience" in parts and p.experience:
        out.append("\n## EXPERIENCE")
        for e in p.experience:
            dates = " - ".join(x for x in (e.start, e.end) if x)
            dur = f", {e.duration_months} months" if e.duration_months else ""
            if computed_years and (span := role_months(e)):
                dur = f", {span[1] - span[0]} months"
            out.append(f"### {e.title} | {e.company} ({dates}{dur})")
            out += [f"- {h}" for h in e.highlights]
    if "projects" in parts and p.projects:
        out.append("\n## PROJECTS")
        for pr in p.projects:
            tech = f" [{', '.join(pr.technologies)}]" if pr.technologies else ""
            out.append(f"- {pr.name}: {pr.description}{tech}")
    if "skills" in parts and p.skills:
        out.append("\n## SKILLS")
        out.append(", ".join(p.skills))
    if "education" in parts and p.education:
        out.append("\n## EDUCATION")
        for ed in p.education:
            fld = ed.field if ed.field and ed.field.lower() not in (ed.degree or "").lower() else None
            out.append("- " + ", ".join(x for x in (ed.degree, fld, ed.institution, ed.year) if x))
    if "certifications" in parts and p.certifications:
        out.append("\n## CERTIFICATIONS")
        out += [f"- {c}" for c in p.certifications]
    if "other" in parts and p.other:
        out.append("\n## OTHER")
        out += [f"- {o}" for o in p.other]
    return "\n".join(out).strip() or "(no relevant resume content for this section)"
