"""API contract tests. None of these reach the LLM: they cover validation, unreadable documents,
error mapping and the LLM-free /rescore endpoint (the pipeline is replaced by a stub)."""
import io

import pytest
from fastapi.testclient import TestClient

import app.api.main as main
from app.core.config import Settings
from app.llm.client import LLMClient, LLMError
from app.core.schemas import ErrorCode
from tests.conftest import make_image_only_pdf, make_text_pdf

JD = ("We are hiring a Senior Backend Engineer. Requirements: 5+ years of Python, PostgreSQL, Kafka, Docker and AWS. "
      "Nice to have: Kubernetes.")


@pytest.fixture
def client(monkeypatch):
    class StubPipeline:
        llm = LLMClient(Settings(), provider="huggingface", model="stub-model", transport=object())

        def run(self, *a, **k):
            raise LLMError(ErrorCode.LLM_UNAVAILABLE, "provider down")
    monkeypatch.setattr(main, "get_pipeline", lambda: StubPipeline())
    monkeypatch.setattr(main, "get_ocr", lambda: None)  # no Tesseract / network in API contract tests
    return TestClient(main.app)


def post(client, resume: tuple, **data):
    return client.post("/api/v1/score", files={"resume": resume}, data=data)


def test_health_and_config(client):
    assert client.get("/health").json()["status"] == "ok"
    cfg = client.get("/api/v1/config").json()
    assert set(cfg["importance_weights"]) == {"must_have", "important", "nice_to_have"}


def test_scanned_resume_returns_structured_422(client):
    r = post(client, ("scan.pdf", make_image_only_pdf(), "application/pdf"), jd_text=JD)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "NO_TEXT_LAYER" and err["field"] == "resume" and err["hint"]


def test_jd_both_or_neither(client, resume_lines):
    cv = ("cv.pdf", make_text_pdf(resume_lines * 3), "application/pdf")
    r = post(client, cv)
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_REQUEST"
    r = client.post("/api/v1/score", files={"resume": cv, "jd_file": ("jd.txt", JD.encode(), "text/plain")},
                    data={"jd_text": JD})
    assert r.status_code == 422


def test_bad_weights(client, resume_lines):
    cv = ("cv.pdf", make_text_pdf(resume_lines * 3), "application/pdf")
    assert post(client, cv, jd_text=JD, weights="{not json").status_code == 422
    assert post(client, cv, jd_text=JD, weights='{"importance_weights": {"must_have": 99}}').status_code == 422


def test_llm_failure_maps_to_503(client, resume_lines):
    cv = ("cv.pdf", make_text_pdf(resume_lines * 3), "application/pdf")
    r = post(client, cv, jd_text=JD)
    assert r.status_code == 503 and r.json()["error"]["code"] == "LLM_UNAVAILABLE"


def test_unsupported_resume_type(client):
    r = post(client, ("cv.png", io.BytesIO(b"\x89PNG\r\n").getvalue(), "image/png"), jd_text=JD)
    assert r.status_code == 422 and r.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_rescore_reweights_without_llm(client):
    crit = {"criterion": {"id": "c1", "name": "Python", "description": "d", "category": "skill",
                          "importance": "must_have", "jd_evidence": ""},
            "raw_level": 3, "final_level": 3, "points": 0, "weight": 0, "weighted_contribution": 0,
            "reasoning": "r", "gaps": "", "evidence": [], "grounded": True, "jd_grounded": True}
    other = {**crit, "criterion": {**crit["criterion"], "id": "c2", "name": "K8s", "importance": "nice_to_have"},
             "raw_level": 0, "final_level": 0}
    base = client.post("/api/v1/rescore", json={"criteria": [crit, other], "overrides": {}}).json()
    zeroed = client.post("/api/v1/rescore", json={"criteria": [crit, other],
                                                   "overrides": {"importance_weights": {"nice_to_have": 0}}}).json()
    assert zeroed["overall_score"] > base["overall_score"]
