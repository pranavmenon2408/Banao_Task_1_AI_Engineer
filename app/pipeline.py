"""Orchestrates: parse -> Agent 1 (profile) -> Agent 2a (criteria, cached per JD) -> Agent 2b (levels)
-> grounding check -> weighted aggregation. Records per-stage latency/tokens and appends each run to
data/runs.jsonl so metrics can be analysed later."""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from app.agents.resume_agent import extract_profile
from app.agents.scoring_agent import extract_criteria, score_profile
from app.cache import JsonCache, text_hash
from app.config import ROOT, ScoringConfig, get_scoring_config, get_settings
from app.grounding import Grounder
from app.llm import LLMClient
from app.prompts import CRITERIA_PROMPT_VERSION, PROMPT_VERSION
from app.schemas import CriteriaList, ResumeProfile, RunMeta, ScoreResult, StageMetric, WeightOverrides
from app.scoring import aggregate, build_scored, strengths_and_gaps

log = logging.getLogger(__name__)


@contextmanager
def _stage(stages: list[StageMetric], name: str):
    m = StageMetric(name=name, latency_ms=0)
    t = time.perf_counter()
    try:
        yield m
    finally:
        m.latency_ms = round((time.perf_counter() - t) * 1000, 1)
        stages.append(m)


class ScoringPipeline:
    def __init__(self, llm: LLMClient | None = None, cfg: ScoringConfig | None = None,
                 cache_criteria: bool = True, cache_profiles: bool = True):
        self.llm = llm or LLMClient()
        self.cfg = cfg or get_scoring_config()
        self.cache_criteria = cache_criteria
        self.cache_profiles = cache_profiles
        self.criteria_cache = JsonCache("criteria")
        self.profile_cache = JsonCache("profiles")

    def get_criteria(self, jd_text: str, stages: list[StageMetric]) -> tuple[CriteriaList, str]:
        key = text_hash(jd_text, self.llm.model, CRITERIA_PROMPT_VERSION, str(self.cfg.limits.max_criteria))
        with _stage(stages, "jd_criteria_extraction") as m:
            cached = self.criteria_cache.get(key) if self.cache_criteria else None
            if cached:
                m.cache_hit = True
                return CriteriaList.model_validate(cached), key
            crit = extract_criteria(self.llm, jd_text, self.cfg.limits.max_criteria, m)
            self.criteria_cache.set(key, crit.model_dump())
            return crit, key

    def get_profile(self, resume_text: str, stages: list[StageMetric]) -> tuple[ResumeProfile, int]:
        key = text_hash(resume_text, self.llm.model, PROMPT_VERSION, str(self.cfg.limits.resume_chunk_tokens))
        with _stage(stages, "resume_profile_extraction") as m:
            cached = self.profile_cache.get(key) if self.cache_profiles else None
            if cached:
                m.cache_hit = True
                return ResumeProfile.model_validate(cached["profile"]), cached["chunks"]
            profile, n = extract_profile(self.llm, resume_text, self.cfg.limits.resume_chunk_tokens, m)
            self.profile_cache.set(key, {"profile": profile.model_dump(), "chunks": n})
            return profile, n

    def run(self, resume_text: str, jd_text: str, overrides: WeightOverrides | None = None,
            warnings: list[str] | None = None) -> ScoreResult:
        t0 = time.perf_counter()
        stages: list[StageMetric] = []
        warnings = list(warnings or [])

        criteria, jd_hash = self.get_criteria(jd_text, stages)
        profile, n_chunks = self.get_profile(resume_text, stages)
        with _stage(stages, "criterion_scoring") as m:
            assessments = score_profile(self.llm, profile, criteria, m, self.cfg.scorer_input_format,
                                        self.cfg.scorer_temperature, self.cfg.scorer_samples)

        with _stage(stages, "grounding_and_aggregation"):
            resume_g, jd_g = Grounder(resume_text), Grounder(jd_text)
            thr = self.cfg.grounding.min_quote_similarity
            scored = []
            for c in criteria.criteria:
                a = assessments[c.id]
                evidence = resume_g.check(a.resume_evidence[:3], thr)
                jd_ok = not c.jd_evidence or jd_g.similarity(c.jd_evidence) >= thr
                scored.append(build_scored(c, a, evidence if a.level > 0 else [], jd_ok, self.cfg))
            overall, label, knockout, weights = aggregate(scored, self.cfg, overrides)
            strengths, gaps = strengths_and_gaps(scored)

        ungrounded = sum(not s.grounded for s in scored)
        if ungrounded:
            warnings.append(f"{ungrounded} criterion score(s) cited evidence not found in the resume and were reduced.")

        meta = RunMeta(run_id=uuid.uuid4().hex[:12], model=self.llm.model, provider=self.llm.provider,
                       total_latency_ms=round((time.perf_counter() - t0) * 1000, 1), stages=stages,
                       resume_chars=len(resume_text), resume_chunks=n_chunks, jd_hash=jd_hash, warnings=warnings)
        result = ScoreResult(overall_score=overall, recommendation=label, knockout_triggered=knockout,
                             role_title=criteria.role_title, candidate_name=profile.candidate_name, criteria=scored,
                             strengths=strengths, gaps=gaps, profile=profile, weights_used=weights, meta=meta)
        self._log_run(result)
        return result

    @staticmethod
    def _log_run(r: ScoreResult) -> None:
        path = Path(get_settings().data_dir)
        path = (path if path.is_absolute() else ROOT / path) / "runs.jsonl"
        row = {"ts": time.time(), "run_id": r.meta.run_id, "model": r.meta.model, "candidate": r.candidate_name,
               "jd_hash": r.meta.jd_hash, "overall": r.overall_score, "levels": {s.criterion.id: s.final_level for s in r.criteria},
               "ungrounded": sum(not s.grounded for s in r.criteria), "latency_ms": r.meta.total_latency_ms,
               "stages": [s.model_dump() for s in r.meta.stages]}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as exc:
            log.warning("Could not write run log: %s", exc)
