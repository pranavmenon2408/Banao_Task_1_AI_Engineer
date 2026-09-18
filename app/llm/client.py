"""Provider-agnostic LLM client: pick a provider and a model (plus fallbacks) and call complete_json.

Why not LangChain: the pipeline is two fixed LLM steps with no tools, memory or routing. What we
actually need (a chat call, JSON parsing, Pydantic validation, retries, metrics) is ~150 lines
here and every step stays visible and debuggable. Providers live in app/providers.py.

Error handling (per model, then down the fallback list):
  * transient failures (timeouts, 429, 5xx, network)  -> exponential backoff retry
  * model not served by the provider                   -> next fallback model; none left -> MODEL_NOT_AVAILABLE
  * bad / missing API key                               -> CONFIG_ERROR immediately
  * provider rejects response_format                    -> prompt-only JSON from then on
  * output isn't valid JSON / fails schema              -> one "repair" turn that shows the model its error
  * retries exhausted on every model                    -> LLM_UNAVAILABLE (a genuinely temporary failure)
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import Settings, get_settings
from app.llm.providers import ProviderError, Transport, build_transport, get_provider
from app.core.schemas import ErrorCode, StageMetric

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


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


def _split(models: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if not models:
        return []
    items = models.split(",") if isinstance(models, str) else models
    return [m.strip() for m in items if m and m.strip()]


class LLMClient:
    """One interface over any provider, with backoff and stepping down to fallback models.

    LLMClient()                                          # text model from .env (LLM_PROVIDER / LLM_MODEL)
    LLMClient(provider="openai", model="gpt-4o-mini")    # explicit
    LLMClient(role="vlm")                                # OCR vision model from .env (VLM_*)
    """

    def __init__(self, settings: Settings | None = None, *, provider: str | None = None, model: str | None = None,
                 fallback_models: str | list[str] | None = None, api_key: str | None = None,
                 endpoint: str | None = None, hf_provider: str | None = None, role: str = "llm",
                 transport: Transport | None = None):
        self.s = settings or get_settings()
        s = self.s
        vlm = role == "vlm"
        self.provider_name = get_provider(provider or (s.vlm_provider if vlm and s.vlm_provider else s.llm_provider)).name
        self.model = model or (s.vlm_model if vlm else s.llm_model)
        fb = fallback_models if fallback_models is not None else (s.vlm_fallback_models if vlm else s.llm_fallback_models)
        self.models = [self.model] + [m for m in _split(fb) if m != self.model]
        self._api_key = api_key if api_key is not None else (s.vlm_api_key if vlm else s.llm_api_key)
        self._endpoint = endpoint if endpoint is not None else (s.vlm_endpoint if vlm else s.llm_endpoint)
        self.hf_provider = hf_provider or (s.vlm_hf_inference_provider if vlm else s.hf_inference_provider)
        self._transport = transport
        self._lock = threading.Lock()
        self._json_mode = True
        self.last_model = self.model        # the model that actually answered the most recent call

    @property
    def provider(self) -> str:
        """Human-readable route, e.g. 'huggingface/auto' or 'openai'."""
        return f"{self.provider_name}/{self.hf_provider}" if self.provider_name == "huggingface" else self.provider_name

    @property
    def transport(self) -> Transport:
        with self._lock:
            if self._transport is None:
                try:
                    self._transport = build_transport(get_provider(self.provider_name), self._api_key or "",
                                                      self._endpoint or "", self.hf_provider)
                except ValueError as exc:
                    raise LLMError(ErrorCode.CONFIG_ERROR, str(exc)) from exc
            return self._transport

    def _call(self, messages: list[dict], max_tokens: int, metric: StageMetric, temperature: float | None = None,
              json_mode: bool = True) -> str:
        temp = self.s.llm_temperature if temperature is None else temperature
        # A fixed seed with temperature > 0 would make every "sample" identical on providers that honour it.
        seed = self.s.llm_seed if temp == 0 else None
        unavailable: list[str] = []
        last: ProviderError | None = None
        for model in self.models:
            attempt = 0
            while True:
                try:
                    metric.llm_calls += 1
                    reply = self.transport.send(model, messages, max_tokens=max_tokens, temperature=temp, seed=seed,
                                                json_mode=json_mode and self._json_mode, timeout=self.s.llm_timeout_s)
                    metric.prompt_tokens += reply.prompt_tokens
                    metric.completion_tokens += reply.completion_tokens
                    if reply.finish_reason == "length":
                        log.warning("LLM output hit max_tokens=%s; JSON may be truncated", max_tokens)
                    if model != self.model:
                        log.warning("Answered by fallback model %s (primary %s unavailable)", model, self.model)
                    self.last_model = model
                    return reply.content
                except ProviderError as exc:
                    last = exc
                    if exc.kind == "auth":
                        raise LLMError(ErrorCode.CONFIG_ERROR,
                                       f"{self.provider_name} rejected the API key ({exc.message[:200]}). "
                                       "Check the key variable for this provider in .env.") from exc
                    if exc.kind == "json_mode" and self._json_mode:
                        log.info("Provider rejected response_format; falling back to prompt-only JSON")
                        self._json_mode = False
                        continue
                    if exc.kind == "transient" and attempt < self.s.llm_max_retries:
                        delay = min(2 ** attempt, 10)
                        log.warning("Transient LLM error on %s (%s); retry %d in %ss", model, exc.message[:120], attempt + 1, delay)
                        attempt += 1
                        metric.retries += 1
                        time.sleep(delay)
                        continue
                    if exc.kind == "model":
                        unavailable.append(model)
                    log.warning("Model %s failed (%s: %s); trying next model if any", model, exc.kind, exc.message[:160])
                    break
        if unavailable and len(unavailable) == len(self.models):
            raise LLMError(ErrorCode.MODEL_NOT_AVAILABLE,
                           f"{', '.join(unavailable)} is not available on provider {self.provider} right now "
                           f"({last.message[:200] if last else ''}).")
        raise LLMError(ErrorCode.LLM_UNAVAILABLE,
                       f"LLM call failed on {', '.join(self.models)} via {self.provider}: {last.message[:300] if last else ''}")

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

    def complete_text(self, messages: list[dict], metric: StageMetric, max_tokens: int = 2000) -> str:
        """Plain-text completion (used for the vision model's page transcription). Same retries and errors."""
        return self._call(messages, max_tokens, metric, temperature=0.0, json_mode=False)
