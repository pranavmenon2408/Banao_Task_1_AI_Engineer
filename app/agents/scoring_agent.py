"""Agent 2: job description -> criteria, then (profile x criteria) -> per-criterion rubric levels.

All criteria are scored in ONE call against the same profile. Scoring each criterion in a separate
call was considered: it isolates criteria from each other, but costs N calls per resume and runs
into provider rate limits. The prompt instead asks the model to rate each criterion independently
and the rubric anchors each level to concrete evidence.

Optional self-consistency: with scorer_samples > 1 the scorer runs that many times in parallel at
scorer_temperature and the median level per criterion wins; the reasoning/evidence shown is taken
from a sample that chose the median level, so explanation and score always agree.
"""
from __future__ import annotations

import re
import statistics
from concurrent.futures import ThreadPoolExecutor

from app.llm import LLMClient, LLMError
from app.render import criteria_text, profile_text
from app.prompts import CRITERIA_EXTRACTOR_SYSTEM, CRITERIA_EXTRACTOR_USER, SCORER_SYSTEM, SCORER_USER
from app.schemas import (AssessmentList, CriteriaList, CriterionAssessment, ErrorCode, ResumeProfile,
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


def score_profile(llm: LLMClient, profile: ResumeProfile, criteria: CriteriaList, metric: StageMetric,
                  input_format: str = "markdown", temperature: float = 0.0,
                  samples: int = 1) -> dict[str, CriterionAssessment]:
    prof = profile_text(profile, input_format)

    def _ask(crits, m: StageMetric) -> dict[str, CriterionAssessment]:
        user = SCORER_USER.format(criteria=criteria_text(crits, input_format), profile=prof)
        res = llm.complete_json(SCORER_SYSTEM, user, AssessmentList, m, max_tokens=3000, temperature=temperature)
        return {a.criterion_id: a for a in res.assessments}

    ids = [c.id for c in criteria.criteria]
    if samples == 1:
        by_id = _ask(criteria.criteria, metric)
    else:
        metrics = [StageMetric(name="sample", latency_ms=0) for _ in range(samples)]
        results, errors = [], []
        with ThreadPoolExecutor(max_workers=samples) as pool:
            futures = [pool.submit(_ask, criteria.criteria, m) for m in metrics]
            for f in futures:
                try:
                    results.append(f.result())
                except LLMError as exc:  # one failed sample shouldn't sink the request
                    errors.append(exc)
        _merge_metric(metric, metrics)
        if not results:
            raise errors[0]
        by_id = _vote(results, ids)

    missing = [cid for cid in ids if cid not in by_id]
    if missing:
        # Rather than silently scoring a missing criterion 0, ask once more for just those.
        retry = _ask([c for c in criteria.criteria if c.id in missing], metric)
        by_id.update({cid: a for cid, a in retry.items() if cid in missing})
    for cid in missing:
        by_id.setdefault(cid, CriterionAssessment(criterion_id=cid, level=0, reasoning="The model returned no assessment for this criterion.",
                                                  resume_evidence=[], gaps="Not assessed."))
    return by_id
