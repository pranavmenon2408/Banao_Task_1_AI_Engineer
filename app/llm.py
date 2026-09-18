"""Thin wrapper around Hugging Face Inference Providers chat completion.

Why not LangChain: the pipeline is two fixed LLM steps with no tools, memory or routing. What we
actually need (a chat call, JSON parsing, Pydantic validation, retries, metrics) is ~150 lines
here and every step stays visible and debuggable.

Error handling:
  * transient failures (timeouts, 429, 5xx, connection resets) -> exponential backoff retry
  * provider rejects response_format -> fall back to prompt-only JSON once, remember it
  * output isn't valid JSON / fails schema -> one "repair" turn that shows the model its error
  * anything else, or retries exhausted -> LLMError with a code the API maps to 502/503
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import TypeVar

import httpx
from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError, HfHubHTTPError, InferenceTimeoutError
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.schemas import ErrorCode, StageMetric

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}


class LLMError(Exception):
    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences and leading chatter."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("top-level JSON value is not an object", text, 0)
    return obj


def _status(exc: Exception) -> int | None:
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None)


class LLMClient:
    def __init__(self, settings: Settings | None = None, client: InferenceClient | None = None,
                 model: str | None = None, provider: str | None = None):
        self.s = settings or get_settings()
        self.model = model or self.s.hf_model
        self.provider = provider or self.s.hf_provider
        self._client = client
        self._json_mode = True

    @property
    def client(self) -> InferenceClient:
        if self._client is None:
            if not self.s.hf_token:
                raise LLMError(ErrorCode.CONFIG_ERROR, "HF_TOKEN is not set. Copy .env.example to .env and add a token.")
            self._client = InferenceClient(provider=self.provider, api_key=self.s.hf_token, timeout=self.s.llm_timeout_s)
        return self._client

    def _call(self, messages: list[dict], max_tokens: int, metric: StageMetric, temperature: float | None = None) -> str:
        attempt = 0
        while True:
            temp = self.s.llm_temperature if temperature is None else temperature
            # A fixed seed with temperature > 0 would make every "sample" identical on providers that honour it.
            kwargs = dict(model=self.model, max_tokens=max_tokens, temperature=temp, seed=self.s.llm_seed if temp == 0 else None)
            if self._json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                metric.llm_calls += 1
                resp = self.client.chat_completion(messages, **kwargs)
                usage = getattr(resp, "usage", None)
                if usage:
                    metric.prompt_tokens += usage.prompt_tokens or 0
                    metric.completion_tokens += usage.completion_tokens or 0
                content = resp.choices[0].message.content or ""
                if resp.choices[0].finish_reason == "length":
                    log.warning("LLM output hit max_tokens=%s; JSON may be truncated", max_tokens)
                return content
            except BadRequestError as exc:
                if self._json_mode and "response_format" in str(exc).lower() + str(getattr(exc, "server_message", "")).lower():
                    log.info("Provider rejected response_format; falling back to prompt-only JSON")
                    self._json_mode = False
                    continue
                raise LLMError(ErrorCode.LLM_UNAVAILABLE, f"LLM rejected the request: {exc}") from exc
            # huggingface_hub >= 1.0 talks HTTP through httpx, so network timeouts surface as
            # httpx.TimeoutException rather than InferenceTimeoutError (observed: a ReadTimeout
            # escaped as an HTTP 500 before this was caught, see docs/DEVLOG.md).
            except (InferenceTimeoutError, HfHubHTTPError, httpx.TransportError, ConnectionError, TimeoutError) as exc:
                status = _status(exc)
                transient = (isinstance(exc, (InferenceTimeoutError, httpx.TransportError, ConnectionError, TimeoutError))
                             or status in TRANSIENT_STATUS)
                if status in (401, 403):
                    raise LLMError(ErrorCode.CONFIG_ERROR, "Hugging Face rejected the token (401/403). Check HF_TOKEN permissions.") from exc
                if not transient or attempt >= self.s.llm_max_retries:
                    raise LLMError(ErrorCode.LLM_UNAVAILABLE, f"LLM call failed after {attempt + 1} attempt(s): {type(exc).__name__}: {exc}") from exc
                delay = min(2 ** attempt, 10)
                log.warning("Transient LLM error (%s, status=%s); retry %d in %ss", type(exc).__name__, status, attempt + 1, delay)
                attempt += 1
                metric.retries += 1
                time.sleep(delay)

    def complete_json(self, system: str, user: str, schema: type[T], metric: StageMetric, max_tokens: int = 2500,
                      temperature: float | None = None) -> T:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        content = self._call(messages, max_tokens, metric, temperature)
        try:
            return schema.model_validate(extract_json(content))
        except (json.JSONDecodeError, ValidationError) as first_err:
            log.warning("LLM output failed validation (%s); attempting one repair turn", type(first_err).__name__)
            metric.retries += 1
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": (
                    "Your previous reply could not be used because it was not valid JSON for the required schema.\n"
                    f"Error: {str(first_err)[:800]}\n"
                    "Reply again with ONLY the corrected JSON object, no prose, no code fences.")},
            ]
            content = self._call(messages, max_tokens, metric, temperature)
            try:
                return schema.model_validate(extract_json(content))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise LLMError(ErrorCode.LLM_BAD_OUTPUT, f"LLM returned invalid output twice: {str(exc)[:300]}") from exc
