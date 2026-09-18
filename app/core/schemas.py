"""Pydantic models shared by the agents, the pipeline, the API and the UI.

Grouped by stage: the resume profile produced by the extraction agent, the criteria and assessments produced by
the scoring agent, and the final result returned by the API.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["skill", "experience", "education", "certification", "domain", "soft_skill"]
Importance = Literal["must_have", "important", "nice_to_have"]


class ErrorCode(StrEnum):
    """Machine-readable error codes returned by the API."""

    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    EMPTY_FILE = "EMPTY_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    CORRUPT_FILE = "CORRUPT_FILE"
    ENCRYPTED_FILE = "ENCRYPTED_FILE"
    NO_TEXT_LAYER = "NO_TEXT_LAYER"
    INSUFFICIENT_TEXT = "INSUFFICIENT_TEXT"
    GARBLED_TEXT = "GARBLED_TEXT"
    INVALID_REQUEST = "INVALID_REQUEST"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_BAD_OUTPUT = "LLM_BAD_OUTPUT"
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    CONFIG_ERROR = "CONFIG_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ApiError(BaseModel):
    """Error body returned by every failing endpoint."""

    code: ErrorCode
    message: str
    hint: str | None = None
    field: str | None = None


# ---------- resume profile (extraction agent) ----------


class ExperienceItem(BaseModel):
    """One role, with its bullets copied verbatim from the resume."""

    title: str = ""
    company: str = ""
    start: str | None = None
    end: str | None = None
    duration_months: int | None = None
    highlights: list[str] = Field(default_factory=list, description="Verbatim bullet text from the resume")


class EducationItem(BaseModel):
    """One degree or qualification."""

    degree: str = ""
    field: str | None = None
    institution: str = ""
    year: str | None = None


class ProjectItem(BaseModel):
    """A project listed on the resume."""

    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    """Structured view of a resume. `summary` is generated; every other field is extracted from the text."""

    candidate_name: str | None = None
    headline: str | None = None
    summary: str = ""
    total_experience_years: float | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list)


# ---------- criteria and assessments (scoring agent) ----------


class Criterion(BaseModel):
    """A single assessable requirement taken from the job description."""

    id: str
    name: str
    description: str
    category: Category
    importance: Importance
    jd_evidence: str = Field("", description="Verbatim phrase from the JD that states this requirement")


class CriteriaList(BaseModel):
    """All criteria extracted from one job description."""

    role_title: str | None = None
    seniority: str | None = None
    criteria: list[Criterion]


class CriterionAssessment(BaseModel):
    """The scorer's judgement for one criterion: a rubric level, its reasoning and quoted evidence."""

    criterion_id: str
    level: int = Field(ge=0, le=4)
    reasoning: str
    resume_evidence: list[str] = Field(default_factory=list)
    gaps: str = ""
    agreement: float = Field(1.0, description="Share of scorer samples that chose this level (1.0 when samples=1)")

    @field_validator("level", mode="before")
    @classmethod
    def _coerce_level(cls, v: Any) -> int:
        return int(round(float(v)))


class AssessmentList(BaseModel):
    """Scorer output: one assessment per criterion."""

    assessments: list[CriterionAssessment]


# ---------- final result ----------


class EvidenceCheck(BaseModel):
    """Whether a quoted piece of evidence was found in the source document."""

    quote: str
    found: bool
    similarity: float


class ScoredCriterion(BaseModel):
    """A criterion with its verified level, points and weighted contribution to the overall score."""

    criterion: Criterion
    raw_level: int
    final_level: int
    points: float
    weight: float
    weighted_contribution: float = Field(description="Points this criterion adds to the overall score")
    reasoning: str
    gaps: str
    evidence: list[EvidenceCheck]
    grounded: bool
    jd_grounded: bool
    agreement: float = 1.0
    flags: list[str] = Field(default_factory=list)


class StageMetric(BaseModel):
    """Latency, LLM usage and cache status of one pipeline stage."""

    name: str
    latency_ms: float
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    cache_hit: bool = False
    detail: str | None = None


class RunMeta(BaseModel):
    """Operational metadata for one scoring run."""

    run_id: str
    model: str
    provider: str
    total_latency_ms: float
    stages: list[StageMetric]
    resume_chars: int
    resume_chunks: int
    jd_hash: str
    extraction: dict[str, str] = Field(default_factory=dict, description="How each document's text was obtained")
    warnings: list[str] = Field(default_factory=list)


class ScoreResult(BaseModel):
    """Full fit assessment returned by `POST /api/v1/score`."""

    overall_score: float = Field(ge=0, le=100)
    recommendation: str
    knockout_triggered: bool
    role_title: str | None = None
    candidate_name: str | None = None
    criteria: list[ScoredCriterion]
    strengths: list[str]
    gaps: list[str]
    profile: ResumeProfile
    weights_used: dict[str, Any]
    meta: RunMeta


class WeightOverrides(BaseModel):
    """Optional per-request override of the configured weights."""

    importance_weights: dict[Importance, float] | None = None
    category_weights: dict[Category, float] | None = None

    @field_validator("importance_weights", "category_weights")
    @classmethod
    def _bounded(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v and any(w < 0 or w > 10 for w in v.values()):
            raise ValueError("weights must be between 0 and 10")
        return v


class RescoreRequest(BaseModel):
    """Existing per-criterion results plus new weights; re-aggregated without calling the LLM."""

    criteria: list[ScoredCriterion]
    overrides: WeightOverrides


class RescoreResult(BaseModel):
    """Result of `POST /api/v1/rescore`."""

    overall_score: float
    recommendation: str
    knockout_triggered: bool
    criteria: list[ScoredCriterion]
    weights_used: dict[str, Any]
