"""FastAPI backend.

    POST /api/v1/score     multipart: resume file + (jd_file | jd_text) [+ weights JSON]  -> ScoreResult
    POST /api/v1/rescore   JSON: existing per-criterion results + new weights (no LLM)    -> RescoreResult
    GET  /api/v1/config    current scoring configuration (weights, bands, limits)
    GET  /health           liveness + whether the LLM is configured

Every error leaves as the same ApiError shape ({code, message, hint, field}) so clients can branch on
`code` instead of parsing messages.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from functools import lru_cache

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, model_validator

from app.core.config import get_scoring_config, get_settings
from app.llm.client import LLMError
from app.documents.ocr import OcrEngine
from app.documents.parsing import DocumentError, check_text, extract_text, normalize
from app.pipeline import ScoringPipeline
from app.core.schemas import (ApiError, ErrorCode, RescoreRequest, RescoreResult, ScoreResult, WeightOverrides)
from app.scoring.aggregation import aggregate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Resume-JD Fit Scorer", version="1.0.0",
              description="Two-agent resume vs job-description fit assessment with per-criterion, evidence-grounded scoring.")

ALLOWED_TYPES = frozenset({"pdf", "docx", "txt"})
LLM_STATUS = {ErrorCode.LLM_UNAVAILABLE: 503, ErrorCode.MODEL_NOT_AVAILABLE: 503, ErrorCode.LLM_BAD_OUTPUT: 502,
              ErrorCode.CONFIG_ERROR: 500}
LLM_HINTS = {
    ErrorCode.LLM_UNAVAILABLE: "The model provider is temporarily failing; retry in a minute.",
    ErrorCode.MODEL_NOT_AVAILABLE: ("Retrying will not help: no provider is serving this model right now. Set LLM_MODEL / "
                                    "LLM_PROVIDER in .env (or add LLM_FALLBACK_MODELS) and restart the API."),
    ErrorCode.CONFIG_ERROR: "Check the provider, model and API key settings in .env, then restart the API.",
}


class ApiException(Exception):
    def __init__(self, status: int, error: ApiError):
        self.status, self.error = status, error


def _err(status: int, code: ErrorCode, message: str, hint: str | None = None, field: str | None = None):
    return ApiException(status, ApiError(code=code, message=message, hint=hint, field=field))


@app.exception_handler(ApiException)
async def _api_exc(_: Request, exc: ApiException):
    return JSONResponse(status_code=exc.status, content={"error": exc.error.model_dump(mode="json")})


@app.exception_handler(RequestValidationError)
async def _validation_exc(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", [])[1:]) or None
    err = ApiError(code=ErrorCode.INVALID_REQUEST, message=first.get("msg", "Invalid request"), field=field)
    return JSONResponse(status_code=422, content={"error": err.model_dump(mode="json")})


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    err = ApiError(code=ErrorCode.INTERNAL_ERROR, message="Unexpected server error. The failure has been logged.",
                   hint="Retry the request; if it persists, check the API logs.")
    return JSONResponse(status_code=500, content={"error": err.model_dump(mode="json")})


@app.middleware("http")
async def _timing(request: Request, call_next):
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
    return ScoringPipeline()


@lru_cache
def get_ocr() -> OcrEngine:
    return OcrEngine(get_scoring_config().ocr)


class ScoreForm(BaseModel):
    """Validates the non-file parts of the multipart form."""
    jd_text: str | None = None
    has_jd_file: bool = False
    weights: WeightOverrides | None = None

    @model_validator(mode="after")
    def _one_jd_source(self):
        has_text = bool(self.jd_text and self.jd_text.strip())
        if has_text == self.has_jd_file:
            raise ValueError("Provide the job description either as jd_text or as jd_file, not both and not neither.")
        if has_text and len(self.jd_text.strip()) < 100:
            raise ValueError("jd_text is too short to extract criteria from (minimum 100 characters).")
        return self


@app.get("/health")
def health():
    llm, ocr = get_pipeline().llm, get_ocr()
    vlm = ocr.vlm if ocr else None
    try:
        configured, problem = bool(llm.transport), None
    except LLMError as exc:
        configured, problem = False, exc.message
    return {"status": "ok", "provider": llm.provider, "model": llm.model, "fallback_models": llm.models[1:],
            "vlm_provider": vlm.provider if vlm else None, "vlm_model": vlm.model if vlm else None,
            "llm_configured": configured, "config_problem": problem}


@app.get("/api/v1/config")
def config():
    return get_scoring_config().model_dump()


@app.post("/api/v1/score", response_model=ScoreResult, responses={
    422: {"model": ApiError, "description": "Unreadable document or invalid request"},
    502: {"model": ApiError, "description": "LLM returned unusable output"},
    503: {"model": ApiError, "description": "LLM provider unavailable"}})
def score(resume: UploadFile = File(..., description="Resume: PDF, DOCX or TXT"),
          jd_file: UploadFile | None = File(None, description="Job description file: PDF, DOCX or TXT"),
          jd_text: str | None = Form(None, description="Job description as plain text"),
          weights: str | None = Form(None, description="Optional JSON WeightOverrides")):
    cfg = get_scoring_config()
    try:
        form = ScoreForm(jd_text=jd_text, has_jd_file=jd_file is not None and bool(jd_file.filename),
                         weights=json.loads(weights) if weights else None)
    except json.JSONDecodeError:
        raise _err(422, ErrorCode.INVALID_REQUEST, "weights must be valid JSON.", field="weights")
    except ValidationError as exc:
        e = exc.errors()[0]
        raise _err(422, ErrorCode.INVALID_REQUEST, e["msg"].removeprefix("Value error, "),
                   field=".".join(map(str, e["loc"])) or "jd")

    warnings: list[str] = []
    try:
        cv = extract_text(resume.filename or "resume", resume.file.read(), allowed=ALLOWED_TYPES,
                          min_chars=cfg.limits.min_resume_chars, max_chars=cfg.limits.max_resume_chars,
                          ocr=get_ocr())
        warnings += cv.warnings
    except DocumentError as exc:
        raise _err(422, exc.code, exc.message, exc.hint, field="resume")
    try:
        if form.has_jd_file:
            jd = extract_text(jd_file.filename, jd_file.file.read(), allowed=ALLOWED_TYPES, min_chars=100,
                              max_chars=cfg.limits.max_jd_chars, purpose="job description", ocr=get_ocr())
        else:
            jd = check_text(normalize(form.jd_text), min_chars=100, max_chars=cfg.limits.max_jd_chars,
                            purpose="job description")
        warnings += jd.warnings
    except DocumentError as exc:
        raise _err(422, exc.code, exc.message, exc.hint, field="jd")

    try:
        return get_pipeline().run(cv.text, jd.text, overrides=form.weights, warnings=warnings,
                                  parse_stages=[m for m in (cv.metric, jd.metric) if m],
                                  extraction={"resume": cv.method, "job_description": jd.method or "text"})
    except LLMError as exc:
        log.error("LLM failure: %s %s", exc.code, exc.message)
        raise _err(LLM_STATUS.get(exc.code, 502), exc.code, exc.message, LLM_HINTS.get(exc.code))


@app.post("/api/v1/rescore", response_model=RescoreResult)
def rescore(req: RescoreRequest):
    """Re-weight without re-running the LLM: weights are configuration, levels are judgements."""
    criteria = [c.model_copy(deep=True) for c in req.criteria]
    if not criteria:
        raise _err(422, ErrorCode.INVALID_REQUEST, "criteria must not be empty.", field="criteria")
    overall, label, knockout, weights = aggregate(criteria, get_scoring_config(), req.overrides)
    return RescoreResult(overall_score=overall, recommendation=label, knockout_triggered=knockout,
                         criteria=criteria, weights_used=weights)
