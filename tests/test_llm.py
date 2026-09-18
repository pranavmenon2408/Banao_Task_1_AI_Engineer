from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings
from app.llm import LLMClient, LLMError
from app.schemas import AssessmentList, ErrorCode, StageMetric


def _resp(content: str):
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat_completion(self, messages, **kw):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _resp(item)


GOOD = '{"assessments": [{"criterion_id": "c1", "level": 3, "reasoning": "r", "resume_evidence": ["Python"]}]}'


def client(script, retries=2):
    s = Settings(hf_token="x", llm_max_retries=retries)
    return LLMClient(settings=s, client=FakeClient(script))


def test_httpx_timeout_is_retried(monkeypatch):
    monkeypatch.setattr("app.llm.time.sleep", lambda _: None)
    llm = client([httpx.ReadTimeout("slow provider"), GOOD])
    m = StageMetric(name="t", latency_ms=0)
    out = llm.complete_json("s", "u", AssessmentList, m)
    assert out.assessments[0].level == 3 and m.retries == 1 and m.llm_calls == 2


def test_retries_exhausted_raise_llm_unavailable(monkeypatch):
    monkeypatch.setattr("app.llm.time.sleep", lambda _: None)
    llm = client([httpx.ConnectError("down")] * 3, retries=2)
    with pytest.raises(LLMError) as e:
        llm.complete_json("s", "u", AssessmentList, StageMetric(name="t", latency_ms=0))
    assert e.value.code == ErrorCode.LLM_UNAVAILABLE


def test_invalid_json_gets_one_repair_turn():
    llm = client(["Sure! here you go", GOOD])
    m = StageMetric(name="t", latency_ms=0)
    assert llm.complete_json("s", "u", AssessmentList, m).assessments[0].criterion_id == "c1"
    assert m.llm_calls == 2


def test_invalid_twice_is_bad_output():
    llm = client(['{"assessments": "nope"}', "still not json"])
    with pytest.raises(LLMError) as e:
        llm.complete_json("s", "u", AssessmentList, StageMetric(name="t", latency_ms=0))
    assert e.value.code == ErrorCode.LLM_BAD_OUTPUT


def test_missing_token_is_config_error():
    llm = LLMClient(settings=Settings(hf_token=""))
    with pytest.raises(LLMError) as e:
        llm.complete_json("s", "u", AssessmentList, StageMetric(name="t", latency_ms=0))
    assert e.value.code == ErrorCode.CONFIG_ERROR
