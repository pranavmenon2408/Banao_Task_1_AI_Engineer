"""End-to-end calibration check against a real LLM provider.

Marked `live`: it needs HF_TOKEN (or the configured provider's key) and is skipped without it. CI runs it as a
separate job using a repository secret. It asserts calibration properties rather than exact scores, because
provider-side inference is not bit-for-bit deterministic.
"""

import os
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.pipeline import ScoringPipeline

pytestmark = pytest.mark.live
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def scores() -> dict[str, float]:
    if not (os.environ.get("HF_TOKEN") or get_settings().llm_api_key):
        if os.environ.get("REQUIRE_LIVE") == "1":  # set in CI so a missing secret fails loudly instead of skipping
            pytest.fail("REQUIRE_LIVE=1 but no LLM credentials are set (HF_TOKEN secret missing?)")
        pytest.skip("no LLM credentials (set HF_TOKEN) - live test skipped")
    jd = (ROOT / "samples/jd_backend_engineer.txt").read_text(encoding="utf-8")
    pipe = ScoringPipeline()
    out = {}
    for name in ("resume_a_strong", "resume_b_strong_similar", "resume_c_adjacent"):
        resume = (ROOT / f"samples/{name}.txt").read_text(encoding="utf-8")
        out[name] = pipe.run(resume, jd).overall_score
    print("live scores:", out)
    return out


def test_similar_resumes_score_close(scores):
    assert abs(scores["resume_a_strong"] - scores["resume_b_strong_similar"]) <= 15


def test_strong_resumes_are_strong_and_weak_resume_is_weak(scores):
    assert scores["resume_a_strong"] >= 70 and scores["resume_b_strong_similar"] >= 70
    assert scores["resume_c_adjacent"] <= 50
    assert min(scores["resume_a_strong"], scores["resume_b_strong_similar"]) - scores["resume_c_adjacent"] >= 30
