"""Agent 2: job description -> criteria, then (profile x criteria) -> per-criterion rubric levels.

All criteria are scored in ONE call against the same profile. Scoring each criterion in a separate
call was considered: it isolates criteria from each other, but costs N calls per resume and runs
into provider rate limits. The prompt instead asks the model to rate each criterion independently
and the rubric anchors each level to concrete evidence.
"""
from __future__ import annotations

import json
import re

from app.llm import LLMClient, LLMError
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


def score_profile(llm: LLMClient, profile: ResumeProfile, criteria: CriteriaList,
                  metric: StageMetric) -> dict[str, CriterionAssessment]:
    crit_json = json.dumps([c.model_dump(exclude={"jd_evidence"}) for c in criteria.criteria], indent=1)
    profile_json = profile.model_dump_json(indent=1, exclude_none=True)
    result = llm.complete_json(SCORER_SYSTEM, SCORER_USER.format(criteria_json=crit_json, profile_json=profile_json),
                               AssessmentList, metric, max_tokens=3000)
    by_id = {a.criterion_id: a for a in result.assessments}
    missing = [c.id for c in criteria.criteria if c.id not in by_id]
    if missing:
        # Rather than silently scoring a missing criterion 0, ask once more for just those.
        subset = CriteriaList(criteria=[c for c in criteria.criteria if c.id in missing])
        retry = llm.complete_json(SCORER_SYSTEM, SCORER_USER.format(
            criteria_json=json.dumps([c.model_dump(exclude={"jd_evidence"}) for c in subset.criteria], indent=1),
            profile_json=profile_json), AssessmentList, metric, max_tokens=2000)
        by_id.update({a.criterion_id: a for a in retry.assessments if a.criterion_id in missing})
    for cid in missing:
        by_id.setdefault(cid, CriterionAssessment(criterion_id=cid, level=0, reasoning="The model returned no assessment for this criterion.",
                                                  resume_evidence=[], gaps="Not assessed."))
    return by_id
