"""API error contract.

Every failure leaves the API with the same body, `{"error": {code, message, hint, field}}`, so clients can branch on
`code` instead of parsing messages. This module maps domain errors to HTTP statuses and registers the handlers.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.schemas import ApiError, ErrorCode
from app.llm.client import LLMError

log = logging.getLogger("api")

LLM_STATUS: dict[ErrorCode, int] = {
    ErrorCode.LLM_UNAVAILABLE: 503,
    ErrorCode.MODEL_NOT_AVAILABLE: 503,
    ErrorCode.LLM_BAD_OUTPUT: 502,
    ErrorCode.CONFIG_ERROR: 500,
}
LLM_HINTS: dict[ErrorCode, str] = {
    ErrorCode.LLM_UNAVAILABLE: "The model provider is temporarily failing; retry in a minute.",
    ErrorCode.MODEL_NOT_AVAILABLE: (
        "Retrying will not help: no provider is serving this model right now. Set LLM_MODEL / "
        "LLM_PROVIDER in .env (or add LLM_FALLBACK_MODELS) and restart the API."
    ),
    ErrorCode.CONFIG_ERROR: "Check the provider, model and API key settings in .env, then restart the API.",
}


class ApiException(Exception):
    """An error to return to the client with the given HTTP status."""

    def __init__(self, status: int, error: ApiError) -> None:
        super().__init__(error.message)
        self.status, self.error = status, error


def api_error(
    status: int, code: ErrorCode, message: str, hint: str | None = None, field: str | None = None
) -> ApiException:
    """Build an ApiException in one call."""
    return ApiException(status, ApiError(code=code, message=message, hint=hint, field=field))


def from_llm_error(exc: LLMError) -> ApiException:
    """Map an LLM failure to its HTTP status and a hint that says whether retrying can help."""
    return api_error(LLM_STATUS.get(exc.code, 502), exc.code, exc.message, LLM_HINTS.get(exc.code))


def _body(error: ApiError) -> dict[str, object]:
    return {"error": error.model_dump(mode="json")}


async def _on_api_exception(_: Request, exc: ApiException) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content=_body(exc.error))


async def _on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", [])[1:]) or None
    err = ApiError(code=ErrorCode.INVALID_REQUEST, message=first.get("msg", "Invalid request"), field=field)
    return JSONResponse(status_code=422, content=_body(err))


async def _on_unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error: %s", exc)
    err = ApiError(
        code=ErrorCode.INTERNAL_ERROR,
        message="Unexpected server error. The failure has been logged.",
        hint="Retry the request; if it persists, check the API logs.",
    )
    return JSONResponse(status_code=500, content=_body(err))


def register_error_handlers(app: FastAPI) -> None:
    """Install the handlers that turn every error into the standard error body."""
    app.add_exception_handler(ApiException, _on_api_exception)
    app.add_exception_handler(RequestValidationError, _on_validation_error)
    app.add_exception_handler(Exception, _on_unhandled)
