"""HTTP API.

    POST /api/v1/score     multipart: resume file + (jd_file | jd_text) [+ weights JSON]  -> ScoreResult
    POST /api/v1/rescore   JSON: existing per-criterion results + new weights (no LLM)    -> RescoreResult
    GET  /api/v1/config    current scoring configuration (weights, bands, limits)
    GET  /health           liveness, and which provider and models are configured

Endpoints are synchronous: FastAPI runs them in a thread pool, which suits the blocking LLM and OCR calls.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from pydantic import BaseModel, ValidationError, model_validator

from app.api.errors import api_error, from_llm_error, register_error_handlers
from app.core.config import get_scoring_config
from app.core.schemas import ApiError, ErrorCode, RescoreRequest, RescoreResult, ScoreResult, WeightOverrides
from app.documents.ocr import OcrEngine
from app.documents.parsing import DocumentError, check_text, extract_text, normalize
from app.llm.client import LLMError
from app.pipeline import ScoringPipeline
from app.scoring.aggregation import aggregate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(
    title="Resume-JD Fit Scorer",
    version="1.0.0",
    description="Two-agent resume vs job-description fit assessment with per-criterion, evidence-grounded scoring.",
)
register_error_handlers(app)

ALLOWED_TYPES = frozenset({"pdf", "docx", "txt"})


@app.middleware("http")
async def _request_timing(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t) * 1000
    response.headers["x-request-id"] = rid
    response.headers["x-latency-ms"] = f"{ms:.0f}"
    log.info("%s %s -> %s in %.0f ms [%s]", request.method, request.url.path, response.status_code, ms, rid)
    return response


@lru_cache
def get_pipeline() -> ScoringPipeline:
    """Shared scoring pipeline (one LLM client and cache per process)."""
    return ScoringPipeline()


@lru_cache
def get_ocr() -> OcrEngine:
    """Shared OCR engine."""
    return OcrEngine(get_scoring_config().ocr)


class ScoreForm(BaseModel):
    """Validates the non-file fields of the scoring form."""

    jd_text: str | None = None
    has_jd_file: bool = False
    weights: WeightOverrides | None = None

    @model_validator(mode="after")
    def _one_jd_source(self) -> ScoreForm:
        has_text = bool(self.jd_text and self.jd_text.strip())
        if has_text == self.has_jd_file:
            raise ValueError("Provide the job description either as jd_text or as jd_file, not both and not neither.")
        if has_text and len(self.jd_text.strip()) < 100:
            raise ValueError("jd_text is too short to extract criteria from (minimum 100 characters).")
        return self


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus the configured provider and models; reports configuration problems without failing."""
    llm, ocr = get_pipeline().llm, get_ocr()
    vlm = ocr.vlm if ocr else None
    try:
        configured, problem = bool(llm.transport), None
    except LLMError as exc:
        configured, problem = False, exc.message
    return {
        "status": "ok",
        "provider": llm.provider,
        "model": llm.model,
        "fallback_models": llm.models[1:],
        "vlm_provider": vlm.provider if vlm else None,
        "vlm_model": vlm.model if vlm else None,
        "llm_configured": configured,
        "config_problem": problem,
    }


@app.get("/api/v1/config")
def config() -> dict[str, Any]:
    """The scoring configuration in effect."""
    return get_scoring_config().model_dump()


@app.post(
    "/api/v1/score",
    response_model=ScoreResult,
    responses={
        422: {"model": ApiError, "description": "Unreadable document or invalid request"},
        502: {"model": ApiError, "description": "LLM returned unusable output"},
        503: {"model": ApiError, "description": "LLM provider or model unavailable"},
    },
)
def score(
    resume: UploadFile = File(..., description="Resume: PDF, DOCX or TXT"),
    jd_file: UploadFile | None = File(None, description="Job description file: PDF, DOCX or TXT"),
    jd_text: str | None = Form(None, description="Job description as plain text"),
    weights: str | None = Form(None, description="Optional JSON WeightOverrides"),
) -> ScoreResult:
    """Score a resume against a job description."""
    cfg = get_scoring_config()
    try:
        form = ScoreForm(
            jd_text=jd_text,
            has_jd_file=jd_file is not None and bool(jd_file.filename),
            weights=json.loads(weights) if weights else None,
        )
    except json.JSONDecodeError as exc:
        raise api_error(422, ErrorCode.INVALID_REQUEST, "weights must be valid JSON.", field="weights") from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        raise api_error(
            422,
            ErrorCode.INVALID_REQUEST,
            first["msg"].removeprefix("Value error, "),
            field=".".join(map(str, first["loc"])) or "jd",
        ) from exc

    warnings: list[str] = []
    try:
        cv = extract_text(
            resume.filename or "resume",
            resume.file.read(),
            allowed=ALLOWED_TYPES,
            min_chars=cfg.limits.min_resume_chars,
            max_chars=cfg.limits.max_resume_chars,
            ocr=get_ocr(),
        )
        warnings += cv.warnings
    except DocumentError as exc:
        raise api_error(422, exc.code, exc.message, exc.hint, field="resume") from exc
    try:
        if form.has_jd_file:
            jd = extract_text(
                jd_file.filename,
                jd_file.file.read(),
                allowed=ALLOWED_TYPES,
                min_chars=100,
                max_chars=cfg.limits.max_jd_chars,
                purpose="job description",
                ocr=get_ocr(),
            )
        else:
            jd = check_text(
                normalize(form.jd_text), min_chars=100, max_chars=cfg.limits.max_jd_chars, purpose="job description"
            )
        warnings += jd.warnings
    except DocumentError as exc:
        raise api_error(422, exc.code, exc.message, exc.hint, field="jd") from exc

    try:
        return get_pipeline().run(
            cv.text,
            jd.text,
            overrides=form.weights,
            warnings=warnings,
            parse_stages=[m for m in (cv.metric, jd.metric) if m],
            extraction={"resume": cv.method, "job_description": jd.method or "text"},
        )
    except LLMError as exc:
        log.error("LLM failure: %s %s", exc.code, exc.message)
        raise from_llm_error(exc) from exc


@app.post("/api/v1/rescore", response_model=RescoreResult)
def rescore(req: RescoreRequest) -> RescoreResult:
    """Re-weight existing per-criterion results without calling the LLM."""
    criteria = [c.model_copy(deep=True) for c in req.criteria]
    if not criteria:
        raise api_error(422, ErrorCode.INVALID_REQUEST, "criteria must not be empty.", field="criteria")
    overall, label, knockout, weights = aggregate(criteria, get_scoring_config(), req.overrides)
    return RescoreResult(
        overall_score=overall,
        recommendation=label,
        knockout_triggered=knockout,
        criteria=criteria,
        weights_used=weights,
    )
