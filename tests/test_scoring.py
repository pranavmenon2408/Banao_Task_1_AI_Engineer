from app.config import get_scoring_config
from app.grounding import Grounder
from app.llm import extract_json
from app.schemas import Criterion, CriterionAssessment, WeightOverrides
from app.scoring import aggregate, build_scored

RESUME = """EXPERIENCE
Senior Engineer, Razorfin (2022 - Present)
- Introduced Kafka-based event pipeline for transaction status updates, cutting reconciliation lag
from 6 hours to 15 minutes.
SKILLS
Python, FastAPI, PostgreSQL"""


def crit(cid, importance="must_have", category="skill"):
    return Criterion(id=cid, name=f"crit {cid}", description="d", category=category, importance=importance)


def scored(cid, level, quotes, importance="must_have", category="skill"):
    cfg = get_scoring_config()
    g = Grounder(RESUME)
    a = CriterionAssessment(criterion_id=cid, level=level, reasoning="r", resume_evidence=quotes)
    ev = g.check(quotes, cfg.grounding.min_quote_similarity) if level else []
    return build_scored(crit(cid, importance, category), a, ev, True, cfg)


def test_grounder_exact_across_line_break():
    g = Grounder(RESUME)
    assert g.similarity("cutting reconciliation lag from 6 hours to 15 minutes") == 1.0


def test_grounder_fuzzy_tolerates_small_differences():
    g = Grounder(RESUME)
    assert g.similarity("Introduced a Kafka based event pipeline for transaction status updates") > 0.85


def test_grounder_accepts_line_assembled_from_real_fragments():
    g = Grounder(RESUME)
    assert g.similarity("Python, FastAPI, PostgreSQL") == 1.0
    assert g.similarity("Senior Engineer | Razorfin") == 1.0
    # ...but one invented fragment sinks the whole quote
    assert g.similarity("Python, FastAPI, Kubernetes") < 0.8


def test_grounder_rejects_hallucination():
    g = Grounder(RESUME)
    assert g.similarity("Familiarity with Kubernetes.") < 0.6
    assert g.similarity("Led a team of 12 engineers") < 0.6


def test_ungrounded_evidence_is_penalised():
    s = scored("c1", 3, ["Deployed services on Kubernetes clusters"])
    assert not s.grounded and s.final_level == 2 and s.flags


def test_grounded_evidence_kept():
    s = scored("c1", 3, ["Python", "Kafka-based event pipeline"])
    assert s.grounded and s.final_level == 3


def test_weighted_average_and_config_weights():
    cfg = get_scoring_config()
    items = [scored("c1", 4, ["Python"], "must_have"), scored("c2", 2, ["FastAPI"], "nice_to_have")]
    overall, _, knockout, _ = aggregate(items, cfg)
    w1, w2 = cfg.importance_weights["must_have"], cfg.importance_weights["nice_to_have"]
    expected = (w1 * cfg.level_points[4] + w2 * cfg.level_points[2]) / (w1 + w2)
    assert overall == round(expected, 1) and not knockout
    assert abs(sum(s.weighted_contribution for s in items) - overall) < 0.1


def test_overrides_change_score_without_llm():
    cfg = get_scoring_config()
    items = [scored("c1", 4, ["Python"], "must_have"), scored("c2", 0, [], "nice_to_have")]
    base, *_ = aggregate(items, cfg)
    boosted, *_ = aggregate(items, cfg, WeightOverrides(importance_weights={"nice_to_have": 0.0}))
    assert boosted == 100.0 and boosted > base


def test_knockout_caps_score():
    cfg = get_scoring_config()
    items = [scored("c1", 4, ["Python"]), scored("c2", 4, ["FastAPI"]), scored("c3", 0, [])]
    overall, label, knockout, _ = aggregate(items, cfg)
    assert knockout and overall <= cfg.knockout.cap


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
