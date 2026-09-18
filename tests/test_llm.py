import json

import httpx
import pytest

from app.core.config import Settings
from app.llm.client import LLMClient, LLMError
from app.llm.providers import OpenAICompatibleTransport, ProviderError, Reply, classify, get_provider
from app.core.schemas import AssessmentList, ErrorCode, StageMetric

GOOD = '{"assessments": [{"criterion_id": "c1", "level": 3, "reasoning": "r", "resume_evidence": ["Python"]}]}'


class FakeTransport:
    """Scripted per-model replies: {model: [reply-or-exception, ...]}."""

    def __init__(self, script: dict[str, list]):
        self.script = {m: list(v) for m, v in script.items()}
        self.calls: list[str] = []

    def send(self, model, messages, **kw):
        self.calls.append(model)
        item = self.script[model].pop(0)
        if isinstance(item, Exception):
            raise item
        return Reply(item, "stop", 10, 5)


def client(script, fallbacks=(), retries=2):
    s = Settings(llm_max_retries=retries)
    return LLMClient(s, provider="huggingface", model="primary", fallback_models=list(fallbacks),
                     transport=FakeTransport(script))


def metric():
    return StageMetric(name="t", latency_ms=0)


def test_transient_error_is_retried(monkeypatch):
    monkeypatch.setattr("app.llm.client.time.sleep", lambda _: None)
    llm = client({"primary": [ProviderError("ReadTimeout", "transient"), GOOD]})
    m = metric()
    assert llm.complete_json("s", "u", AssessmentList, m).assessments[0].level == 3
    assert m.retries == 1 and m.llm_calls == 2 and m.prompt_tokens == 10


def test_retries_exhausted_is_temporary_unavailable(monkeypatch):
    monkeypatch.setattr("app.llm.client.time.sleep", lambda _: None)
    llm = client({"primary": [ProviderError("503", "transient", 503)] * 3}, retries=2)
    with pytest.raises(LLMError) as e:
        llm.complete_json("s", "u", AssessmentList, metric())
    assert e.value.code == ErrorCode.LLM_UNAVAILABLE


def test_model_not_supported_is_not_retried_and_says_so():
    """The observed failure: HF 'model_not_supported' used to be reported as 'temporary, retry in a minute'."""
    llm = client({"primary": [ProviderError("model_not_supported", "model", 400)]})
    with pytest.raises(LLMError) as e:
        llm.complete_json("s", "u", AssessmentList, metric())
    assert e.value.code == ErrorCode.MODEL_NOT_AVAILABLE and llm.transport.calls == ["primary"]


def test_steps_down_to_fallback_model_and_reports_it():
    llm = client({"primary": [ProviderError("model_not_supported", "model", 400)], "backup": [GOOD]},
                 fallbacks=["backup"])
    llm.complete_json("s", "u", AssessmentList, metric())
    assert llm.transport.calls == ["primary", "backup"] and llm.last_model == "backup"


def test_bad_key_is_config_error_without_fallback():
    llm = client({"primary": [ProviderError("401", "auth", 401)], "backup": [GOOD]}, fallbacks=["backup"])
    with pytest.raises(LLMError) as e:
        llm.complete_json("s", "u", AssessmentList, metric())
    assert e.value.code == ErrorCode.CONFIG_ERROR and llm.transport.calls == ["primary"]


def test_invalid_json_gets_one_repair_turn():
    llm = client({"primary": ["Sure! here you go", GOOD]})
    m = metric()
    assert llm.complete_json("s", "u", AssessmentList, m).assessments[0].criterion_id == "c1"
    assert m.llm_calls == 2


def test_invalid_twice_is_bad_output():
    llm = client({"primary": ['{"assessments": "nope"}', "still not json"]})
    with pytest.raises(LLMError) as e:
        llm.complete_json("s", "u", AssessmentList, metric())
    assert e.value.code == ErrorCode.LLM_BAD_OUTPUT


def test_missing_key_is_config_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = LLMClient(Settings(), provider="openai", model="gpt-4o-mini")
    with pytest.raises(LLMError) as e:
        llm.complete_json("s", "u", AssessmentList, metric())
    assert e.value.code == ErrorCode.CONFIG_ERROR and "OPENAI_API_KEY" in e.value.message


def test_provider_registry_and_aliases():
    assert get_provider("gemini").name == "google" and get_provider("HF").name == "huggingface"
    with pytest.raises(ValueError):
        get_provider("nope")


@pytest.mark.parametrize("status,text,kind", [
    (401, "bad key", "auth"), (404, "model_not_found", "model"), (400, "model_not_supported", "model"),
    (429, "slow down", "transient"), (None, "ReadTimeout", "transient"), (400, "bad response_format", "json_mode"),
    (400, "max_tokens too large", "other")])
def test_error_classification(status, text, kind):
    assert classify(status, text) == kind


def _mock_openai(handler) -> OpenAICompatibleTransport:
    return OpenAICompatibleTransport("https://api.example.com/v1/chat/completions", "sk-test",
                                     client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_openai_compatible_transport_request_and_usage():
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": GOOD}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 12, "completion_tokens": 7}})

    reply = _mock_openai(handler).send("gpt-4o-mini", [{"role": "user", "content": "hi"}], max_tokens=50,
                                       temperature=0, seed=42, json_mode=True, timeout=5)
    assert reply.content == GOOD and (reply.prompt_tokens, reply.completion_tokens) == (12, 7)
    assert seen["auth"] == "Bearer sk-test" and seen["body"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_transport_maps_errors():
    t = _mock_openai(lambda r: httpx.Response(404, json={"error": {"code": "model_not_found"}}))
    with pytest.raises(ProviderError) as e:
        t.send("nope", [], max_tokens=5, temperature=0, seed=None, json_mode=False, timeout=5)
    assert e.value.kind == "model"
    t = _mock_openai(lambda r: httpx.Response(429, text="rate limited"))
    with pytest.raises(ProviderError) as e:
        t.send("m", [], max_tokens=5, temperature=0, seed=None, json_mode=False, timeout=5)
    assert e.value.kind == "transient"


def test_openai_compatible_transport_drops_rejected_seed():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "seed" in body:
            return httpx.Response(400, text="Unrecognized request argument supplied: seed")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    assert _mock_openai(handler).send("m", [], max_tokens=5, temperature=0, seed=42, json_mode=False,
                                      timeout=5).content == "ok"
    assert "seed" in bodies[0] and "seed" not in bodies[1]
