"""Turn per-criterion rubric levels into the overall score.

Aggregation is plain code driven entirely by `config/scoring.yaml`: the LLM only assigns a level per criterion, so
changing weights never changes a judgement and the arithmetic is reproducible.

    weight_i  = importance_weights[importance_i] * category_weights[category_i]
    points_i  = level_points[final_level_i]          (final level = raw level minus any grounding penalty)
    overall   = sum(weight_i * points_i) / sum(weight_i)
    if knockout is enabled and a must-have criterion is at level 0: overall = min(overall, knockout.cap)
"""

from __future__ import annotations

from app.core.config import ScoringConfig
from app.core.schemas import Criterion, CriterionAssessment, EvidenceCheck, ScoredCriterion, WeightOverrides


def effective_weights(cfg: ScoringConfig, overrides: WeightOverrides | None) -> tuple[dict, dict]:
    """Configured importance and category weights with per-request overrides applied."""
    imp = dict(cfg.importance_weights)
    cat = dict(cfg.category_weights)
    if overrides:
        imp.update(overrides.importance_weights or {})
        cat.update(overrides.category_weights or {})
    return imp, cat


def build_scored(
    criterion: Criterion, a: CriterionAssessment, evidence: list[EvidenceCheck], jd_grounded: bool, cfg: ScoringConfig
) -> ScoredCriterion:
    """Apply grounding results to an assessment: lower the level if its evidence is not in the resume."""
    flags: list[str] = []
    grounded = a.level == 0 or any(e.found for e in evidence)
    final = a.level
    if not grounded:
        final = max(0, a.level - cfg.grounding.ungrounded_level_penalty)
        flags.append(
            "Evidence quote not found in resume; level reduced"
            if evidence
            else "No evidence quoted for a non-zero level; level reduced"
        )
    elif any(not e.found for e in evidence):
        flags.append("Some quoted evidence was not found verbatim in the resume")
    if a.agreement < 0.6:
        flags.append(f"Low scorer agreement ({a.agreement:.0%} of samples chose this level)")
    if not jd_grounded:
        flags.append("Requirement phrase not found verbatim in the job description")
    return ScoredCriterion(
        criterion=criterion,
        raw_level=a.level,
        final_level=final,
        points=0,
        weight=0,
        weighted_contribution=0,
        reasoning=a.reasoning,
        gaps=a.gaps,
        evidence=evidence,
        grounded=grounded,
        jd_grounded=jd_grounded,
        agreement=a.agreement,
        flags=flags,
    )


def aggregate(
    scored: list[ScoredCriterion], cfg: ScoringConfig, overrides: WeightOverrides | None = None
) -> tuple[float, str, bool, dict]:
    """Weight and combine levels; returns (overall score, recommendation, knockout applied, weights used)."""
    imp, cat = effective_weights(cfg, overrides)
    for s in scored:
        s.weight = imp.get(s.criterion.importance, 1.0) * cat.get(s.criterion.category, 1.0)
        s.points = cfg.level_points[s.final_level]
    total_w = sum(s.weight for s in scored)
    if total_w <= 0:
        overall = 0.0
        for s in scored:
            s.weighted_contribution = 0.0
    else:
        for s in scored:
            s.weighted_contribution = round(s.weight * s.points / total_w, 2)
        overall = sum(s.weight * s.points for s in scored) / total_w

    knockout = cfg.knockout.enabled and any(
        s.criterion.importance == "must_have" and s.final_level == 0 for s in scored
    )
    if knockout:
        overall = min(overall, cfg.knockout.cap)
    overall = round(overall, 1)
    label = next(b.label for b in cfg.recommendation_bands if overall >= b.min)
    return (
        overall,
        label,
        knockout,
        {
            "importance_weights": imp,
            "category_weights": cat,
            "level_points": cfg.level_points,
            "knockout": cfg.knockout.model_dump(),
        },
    )


def strengths_and_gaps(scored: list[ScoredCriterion]) -> tuple[list[str], list[str]]:
    """Summarise the strongest criteria and the most important gaps for the recruiter."""
    strengths = [
        f"{s.criterion.name} (level {s.final_level}/4)"
        for s in sorted(scored, key=lambda s: (-s.final_level, -s.weight))
        if s.final_level >= 3
    ][:5]
    gaps = [
        f"{s.criterion.name}: {s.gaps or 'no evidence in resume'}"
        for s in sorted(scored, key=lambda s: (s.final_level, -s.weight))
        if s.final_level <= 1 and s.criterion.importance != "nice_to_have"
    ][:5]
    return strengths, gaps
