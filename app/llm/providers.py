"""LLM providers: where a request goes, how it authenticates, and how it is sent.

Any provider with an OpenAI-compatible /chat/completions API works through one transport; Hugging Face
keeps its own transport so its inference-provider routing ("auto", "featherless-ai", ...) still works.
Every failure is raised as a ProviderError with a `kind`, so the client can react correctly:

    auth         bad or missing key                 -> stop, configuration error
    model        model not served by this provider  -> try the next fallback model, else configuration error
    transient    timeout, 429, 5xx, network         -> back off and retry
    other        any other rejected request         -> try the next fallback model
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

Kind = Literal["auth", "model", "transient", "json_mode", "other"]
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
_MODEL_MISSING_HINTS = ("model_not_supported", "not supported by provider", "not supported by any provider",
                        "model_not_found", "does not exist", "unknown model", "no endpoints found", "invalid model")


class ProviderError(Exception):
    def __init__(self, message: str, kind: Kind, status: int | None = None):
        super().__init__(message)
        self.message, self.kind, self.status = message, kind, status


def classify(status: int | None, text: str) -> Kind:
    low = text.lower()
    if status in (401, 403):
        return "auth"
    if any(h in low for h in _MODEL_MISSING_HINTS) or status == 404:
        return "model"
    if status == 400 and "response_format" in low:
        return "json_mode"
    if status is None or status in TRANSIENT_STATUS:
        return "transient"
    return "other"


@dataclass
class Reply:
    content: str
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Transport(Protocol):
    def send(self, model: str, messages: list[dict[str, Any]], *, max_tokens: int, temperature: float,
             seed: int | None, json_mode: bool, timeout: float) -> Reply: ...


@dataclass(frozen=True)
class Provider:
    name: str
    endpoint: str = ""
    key_env: tuple[str, ...] = ()
    key_required: bool = True


PROVIDERS: dict[str, Provider] = {p.name: p for p in (
    Provider("huggingface", key_env=("HF_TOKEN", "HF_API_KEY")),
    Provider("openai", "https://api.openai.com/v1/chat/completions", ("OPENAI_API_KEY",)),
    Provider("google", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
             ("GOOGLE_API_KEY", "GEMINI_API_KEY")),
    Provider("groq", "https://api.groq.com/openai/v1/chat/completions", ("GROQ_API_KEY",)),
    Provider("mistral", "https://api.mistral.ai/v1/chat/completions", ("MISTRAL_API_KEY",)),
    Provider("openrouter", "https://openrouter.ai/api/v1/chat/completions", ("OPENROUTER_API_KEY",)),
    Provider("together", "https://api.together.xyz/v1/chat/completions", ("TOGETHER_API_KEY",)),
    Provider("ollama", "http://localhost:11434/v1/chat/completions", ("OLLAMA_API_KEY",), key_required=False),
    Provider("openai-compatible", key_env=("LLM_API_KEY",), key_required=False),   # any other endpoint
)}
_ALIASES = {"hf": "huggingface", "gemini": "google", "custom": "openai-compatible", "local": "ollama"}


def get_provider(name: str) -> Provider:
    key = _ALIASES.get(name.strip().lower(), name.strip().lower())
    if key not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {name!r}; expected one of {', '.join(PROVIDERS)}")
    return PROVIDERS[key]


def resolve_api_key(provider: Provider, explicit: str = "") -> str:
    return explicit or next((os.environ[n] for n in provider.key_env if os.environ.get(n)), "")


class OpenAICompatibleTransport:
    def __init__(self, endpoint: str, api_key: str, client: httpx.Client | None = None):
        self.endpoint, self.api_key = endpoint, api_key
        self.client = client or httpx.Client()
        self._send_seed = True

    def send(self, model, messages, *, max_tokens, temperature, seed, json_mode, timeout) -> Reply:
        body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens,
                                "temperature": temperature}
        if seed is not None and self._send_seed:
            body["seed"] = seed
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = self.client.post(self.endpoint, json=body, headers=headers, timeout=timeout)
        except httpx.TransportError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}", "transient") from exc
        if r.status_code >= 400:
            text = r.text[:500]
            if r.status_code == 400 and "seed" in text.lower() and self._send_seed:
                self._send_seed = False     # some providers reject `seed`; drop it and retry once
                return self.send(model, messages, max_tokens=max_tokens, temperature=temperature, seed=None,
                                 json_mode=json_mode, timeout=timeout)
            raise ProviderError(f"HTTP {r.status_code}: {text}", classify(r.status_code, text), r.status_code)
        try:
            data = r.json()
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            return Reply(choice["message"].get("content") or "", str(choice.get("finish_reason") or ""),
                         int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {r.text[:300]}", "other", r.status_code) from exc


class HuggingFaceTransport:
    """huggingface_hub InferenceClient; `hf_provider` picks the inference provider ("auto" = first enabled)."""

    def __init__(self, api_key: str, hf_provider: str = "auto"):
        from huggingface_hub import InferenceClient
        self.hf_provider = hf_provider
        self.client = InferenceClient(provider=hf_provider, api_key=api_key)

    def send(self, model, messages, *, max_tokens, temperature, seed, json_mode, timeout) -> Reply:
        from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError
        kwargs: dict[str, Any] = dict(model=model, max_tokens=max_tokens, temperature=temperature, seed=seed)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self.client.timeout = timeout
        try:
            resp = self.client.chat_completion(messages, **kwargs)
        # huggingface_hub >= 1.0 uses httpx, so network timeouts arrive as httpx errors, not
        # InferenceTimeoutError (observed as an unhandled HTTP 500, docs/DEVLOG.md entry 2).
        except (InferenceTimeoutError, httpx.TransportError, ConnectionError, TimeoutError) as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}", "transient") from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            text = f"{exc} {getattr(exc, 'server_message', '')}"
            raise ProviderError(text[:600], classify(status, text), status) from exc
        except ValueError as exc:  # raised client-side, e.g. "Model X is not supported by provider Y"
            raise ProviderError(str(exc)[:600], classify(None, str(exc)) if "support" in str(exc) else "other") from exc
        usage = getattr(resp, "usage", None)
        return Reply(resp.choices[0].message.content or "", str(resp.choices[0].finish_reason or ""),
                     getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)


def build_transport(provider: Provider, api_key: str = "", endpoint: str = "", hf_provider: str = "auto") -> Transport:
    key = resolve_api_key(provider, api_key)
    if provider.key_required and not key:
        raise ValueError(f"LLM provider {provider.name!r} needs an API key: set {' or '.join(provider.key_env)}"
                         " (or LLM_API_KEY)")
    if provider.name == "huggingface":
        return HuggingFaceTransport(key, hf_provider)
    url = endpoint or provider.endpoint
    if not url:
        raise ValueError(f"LLM provider {provider.name!r} needs an endpoint: set LLM_ENDPOINT")
    return OpenAICompatibleTransport(url, key)
