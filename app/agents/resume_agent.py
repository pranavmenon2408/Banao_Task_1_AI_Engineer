"""Extraction agent: raw resume text -> structured `ResumeProfile`.

A typical resume is extracted in a single call. Long resumes are split at section boundaries, each chunk is
extracted independently, and the partial profiles are merged in code. Merging deterministically, rather than with
another LLM call, keeps the verbatim bullet text that quote verification depends on.
"""

from __future__ import annotations

from datetime import date

from app.agents.prompts import CHUNK_NOTE, RESUME_EXTRACTOR_SYSTEM, RESUME_EXTRACTOR_USER
from app.core.schemas import ResumeProfile, StageMetric
from app.documents.chunking import chunk_resume
from app.llm.client import LLMClient


def _dedupe(items: list[str]) -> list[str]:
    """Remove case/whitespace-insensitive duplicates, keeping first occurrences."""
    seen, out = set(), []
    for s in items:
        k = " ".join(s.lower().split())
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


def merge_profiles(parts: list[ResumeProfile]) -> ResumeProfile:
    """Merge per-chunk profiles; a role that spans two chunks is joined rather than duplicated."""
    if len(parts) == 1:
        return parts[0]
    merged = ResumeProfile()
    for p in parts:
        merged.candidate_name = merged.candidate_name or p.candidate_name
        merged.headline = merged.headline or p.headline
        if p.summary and len(p.summary) > len(merged.summary):
            merged.summary = p.summary
        if p.total_experience_years is not None:
            merged.total_experience_years = max(merged.total_experience_years or 0, p.total_experience_years)
        merged.skills += p.skills
        merged.education += p.education
        merged.projects += p.projects
        merged.certifications += p.certifications
        merged.other += p.other
        for exp in p.experience:
            # A role split across chunks shows up twice: join its highlights instead of duplicating it.
            same = next(
                (
                    e
                    for e in merged.experience
                    if e.title.lower() == exp.title.lower() and e.company.lower() == exp.company.lower()
                ),
                None,
            )
            if same:
                same.highlights = _dedupe(same.highlights + exp.highlights)
                same.start, same.end = same.start or exp.start, same.end or exp.end
                same.duration_months = same.duration_months or exp.duration_months
            else:
                merged.experience.append(exp.model_copy(deep=True))
    merged.skills = _dedupe(merged.skills)
    merged.certifications = _dedupe(merged.certifications)
    merged.other = _dedupe(merged.other)
    return merged


def extract_profile(
    llm: LLMClient, resume_text: str, chunk_tokens: int, metric: StageMetric
) -> tuple[ResumeProfile, int]:
    """Extract a profile from resume text; returns (profile, number of chunks used)."""
    chunks = chunk_resume(resume_text, chunk_tokens)
    system = RESUME_EXTRACTOR_SYSTEM.format(today=date.today().strftime("%b %Y"))
    parts = []
    for i, chunk in enumerate(chunks, 1):
        note = CHUNK_NOTE.format(i=i, n=len(chunks)) if len(chunks) > 1 else ""
        parts.append(
            llm.complete_json(
                system,
                RESUME_EXTRACTOR_USER.format(chunk_note=note, text=chunk),
                ResumeProfile,
                metric,
                max_tokens=3500,
            )
        )
    return merge_profiles(parts), len(chunks)
