"""Agent 2: job description -> criteria, then (profile x criteria) -> per-criterion rubric levels.

Two scoring modes (config: scoring_mode):
  single     all criteria in ONE call against the whole profile. Cheapest, but in calibration runs
             the model judged criteria together: c2-c5 flipped 3<->4 as a block (halo effect).
  sectioned  criteria are grouped by JD section (experience / skills / education / domain) and each
             group is scored in its own call, in parallel, against only the resume sections that can
             evidence it, with section-specific rules (e.g. a skill named only in the skills list is
             at most "partial"; years of experience are computed in code, not by the model).
Scoring every criterion in its own call was also considered; sections are the middle ground between
isolation and the number of calls (~4 vs ~10 per resume) under provider rate limits.

Optional self-consistency: with scorer_samples > 1 the scorer runs that many times in parallel at
scorer_temperature and the median level per criterion wins; the reasoning/evidence shown is taken
from a sample that chose the median level, so explanation and score always agree.
"""
from __future__ import annotations

import re
import statistics
from concurrent.futures import ThreadPoolExecutor

from app.llm.client import LLMClient, LLMError
from app.agents.render import ALL_PARTS, criteria_text, profile_text
from app.agents.prompts import (CATEGORY_SECTION, CRITERIA_EXTRACTOR_SYSTEM, CRITERIA_EXTRACTOR_USER, SCORER_SYSTEM,
                         SCORER_USER, SECTION_ALL, SECTION_FOCUS, SECTION_ONLY, SECTION_PARTS)
from app.core.schemas import (AssessmentList, CriteriaList, CriterionAssessment, ErrorCode, ResumeProfile,
                         StageMetric)

IMPORTANCE_ORDER = {"must_have": 0, "important": 1, "nice_to_have": 2}


def extract_criteria(llm: LLMClient, jd_text: str, max_criteria: int, metric: StageMetric) -> CriteriaList:
    result = llm.complete_json(CRITERIA_EXTRACTOR_SYSTEM.format(max_criteria=max_criteria),
                               CRITERIA_EXTRACTOR_USER.format(jd=jd_text), CriteriaList, metric, max_tokens=2500)
    if not result.criteria:
        raise LLMError(ErrorCode.LLM_BAD_OUTPUT, "No assessable criteria could be extracted from the job description.")
    # Normalise: stable order, unique ids, cap count. Deterministic post-processing keeps the rubric stable.
    crit = sorted(result.criteria, key=lambda c: IMPORTANCE_ORDER[c.importance])[:max_criteria]
    seen: set[str] = set()
    unique = []
    for c in crit:
        key = re.sub(r"\W+", " ", c.name.lower()).strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    for i, c in enumerate(unique, 1):
        c.id = f"c{i}"
    result.criteria = unique
    return result


def _merge_metric(into: StageMetric, parts: list[StageMetric]) -> None:
    for m in parts:
        into.llm_calls += m.llm_calls
        into.prompt_tokens += m.prompt_tokens
        into.completion_tokens += m.completion_tokens
        into.retries += m.retries


def _vote(samples: list[dict[str, CriterionAssessment]], ids: list[str]) -> dict[str, CriterionAssessment]:
    out = {}
    for cid in ids:
        cands = [s[cid] for s in samples if cid in s]
        if not cands:
            continue
        levels = [c.level for c in cands]
        med = int(statistics.median_low(levels))
        chosen = next(c for c in cands if c.level == med).model_copy()
        chosen.agreement = round(levels.count(med) / len(levels), 2)
        out[cid] = chosen
    return out


def _jobs(profile: ResumeProfile, criteria: list, mode: str, fmt: str) -> list[tuple[str, str, list]]:
    """Return (system prompt, profile text, criteria) per scoring call."""
    groups: dict[str, list] = {}
    for c in criteria:
        groups.setdefault(CATEGORY_SECTION.get(c.category, "domain"), []).append(c)
    if mode == "single":
        rules = "\n\n".join(SECTION_FOCUS[sec] for sec in groups)
        return [(SCORER_SYSTEM + "\n\n" + SECTION_ALL + rules, profile_text(profile, fmt, computed_years=True), criteria)]
    return [(SCORER_SYSTEM + "\n\n" + SECTION_ONLY + SECTION_FOCUS[sec],
             profile_text(profile, fmt, SECTION_PARTS.get(sec, ALL_PARTS), computed_years=True),
             crits) for sec, crits in groups.items()]


def score_profile(llm: LLMClient, profile: ResumeProfile, criteria: CriteriaList, metric: StageMetric,
                  input_format: str = "markdown", temperature: float = 0.0, samples: int = 1,
                  mode: str = "single") -> dict[str, CriterionAssessment]:
    def _ask(system: str, prof: str, crits: list, m: StageMetric) -> dict[str, CriterionAssessment]:
        user = SCORER_USER.format(criteria=criteria_text(crits, input_format), profile=prof)
        res = llm.complete_json(system, user, AssessmentList, m, max_tokens=3000, temperature=temperature)
        wanted = {c.id for c in crits}
        return {a.criterion_id: a for a in res.assessments if a.criterion_id in wanted}

    jobs = _jobs(profile, criteria.criteria, mode, input_format)
    calls = [(j, k) for j in range(len(jobs)) for k in range(samples)]
    metrics = [StageMetric(name="call", latency_ms=0) for _ in calls]
    per_job: list[list[dict]] = [[] for _ in jobs]
    errors: list[LLMError] = []
    with ThreadPoolExecutor(max_workers=min(8, len(calls))) as pool:
        futures = [pool.submit(_ask, *jobs[j], m) for (j, _), m in zip(calls, metrics)]
        for (j, _), f in zip(calls, futures):
            try:
                per_job[j].append(f.result())
            except LLMError as exc:  # one failed call shouldn't sink the request; missing ids are retried below
                errors.append(exc)
    _merge_metric(metric, metrics)
    if errors and not any(per_job):
        raise errors[0]

    by_id: dict[str, CriterionAssessment] = {}
    for (_, _, crits), results in zip(jobs, per_job):
        by_id.update(_vote(results, [c.id for c in crits]))

    missing = [c for c in criteria.criteria if c.id not in by_id]
    if missing:
        # Rather than silently scoring a missing criterion 0, ask once more for just those.
        for system, prof, crits in _jobs(profile, missing, mode, input_format):
            by_id.update(_ask(system, prof, crits, metric))
    for c in criteria.criteria:
        by_id.setdefault(c.id, CriterionAssessment(criterion_id=c.id, level=0, reasoning="The model returned no assessment for this criterion.",
                                                   resume_evidence=[], gaps="Not assessed."))
    return by_id
