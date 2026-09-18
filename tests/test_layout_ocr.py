import io
from pathlib import Path

import pymupdf
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app import pdf_layout
from app.config import OcrConfig
from app.ocr import OcrEngine, OcrFailed
from app.parsing import DocumentError, extract_text
from app.schemas import ErrorCode, StageMetric
from scripts.make_samples import scanned_image_pdf, two_column_pdf
from tests.conftest import make_image_only_pdf

ROOT = Path(__file__).resolve().parent.parent
RESUME_A = (ROOT / "samples/resume_a_strong.txt").read_text(encoding="utf-8").splitlines()


def _doc(data: bytes) -> pymupdf.Document:
    return pymupdf.open(stream=data, filetype="pdf")


# ---------------------------------------------------------------- multi-column

def test_two_column_is_read_column_by_column():
    left = ["SKILLS", "Python", "Kafka", "Docker", "Redis", "AWS", "EDUCATION", "B.E. Computer Science"]
    right = ["EXPERIENCE", "Senior Engineer, Razorfin (2022 - Present)",
             "- Introduced Kafka-based event pipeline for transaction status updates, cutting lag",
             "- Optimised PostgreSQL queries and partitioned the ledger table, reducing latency",
             "- Mentored three junior engineers and led design reviews", "Engineer, LendQuick (2019 - 2022)",
             "- Built loan-disbursal microservices with Django"]
    text, multi = pdf_layout.extract(_doc(two_column_pdf(left, right)))
    assert multi
    lines = text.splitlines()
    # the whole sidebar comes before the main column, and no sidebar word lands inside a bullet
    assert lines.index("B.E. Computer Science") < lines.index("EXPERIENCE")
    joined = " ".join(lines[lines.index("EXPERIENCE"):])
    assert "cutting lag" in joined and "Redis" not in joined


def test_right_aligned_dates_are_not_a_second_column():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for i in range(6):
        c.drawString(50, y, f"Senior Engineer {i}, Company {i}")
        c.drawRightString(545, y, f"201{i} - 201{i + 1}")
        y -= 14
        for b in range(3):
            c.drawString(60, y, f"- Built service {i}.{b} handling millions of requests for merchants every day")
            y -= 14
    c.save()
    text, multi = pdf_layout.extract(_doc(buf.getvalue()))
    assert not multi
    assert "Senior Engineer 3, Company 3  2013 - 2014" in text   # date stays on its role's line


# ---------------------------------------------------------------- OCR chain (fakes, no network)

class FakeVLM:
    last_model = "fake/vision-model"

    def __init__(self, text="", fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def complete_text(self, messages, metric, max_tokens=2000):
        from app.llm import LLMError
        self.calls += 1
        metric.llm_calls += 1
        metric.prompt_tokens += 2500
        if self.fail:
            raise LLMError(ErrorCode.LLM_UNAVAILABLE, "down")
        return self.text


def engine(tesseract_result=None, vlm=None, **cfg) -> OcrEngine:
    e = OcrEngine(OcrConfig(**cfg), vlm=vlm)
    if tesseract_result is None:
        e.__dict__["tesseract"] = None           # simulate "binary not installed"
    else:
        e.__dict__["tesseract"] = object()
        e._tesseract_page = lambda page: tesseract_result
    return e


GOOD_TEXT = "Senior Engineer at Razorfin Payments. Built Kafka pipelines and PostgreSQL ledgers. " * 5


def test_tesseract_confident_result_is_used_without_vlm():
    vlm = FakeVLM(GOOD_TEXT)
    out = engine((GOOD_TEXT, 92.0 * len(GOOD_TEXT), len(GOOD_TEXT)), vlm).run(_doc(make_image_only_pdf()), StageMetric(name="t", latency_ms=0))
    assert out.method == "tesseract" and out.confidence == 92.0 and vlm.calls == 0


def test_low_tesseract_confidence_falls_back_to_vlm_and_counts_tokens():
    vlm = FakeVLM(GOOD_TEXT)
    m = StageMetric(name="t", latency_ms=0)
    out = engine(("r4z0rf1n", 35.0 * 8, 8), vlm).run(_doc(make_image_only_pdf(2)), m)
    assert out.method == "vlm" and vlm.calls == 2 and m.prompt_tokens == 5000
    assert any("confidence" in n for n in out.notes)


def test_missing_tesseract_goes_straight_to_vlm():
    out = engine(None, FakeVLM(GOOD_TEXT)).run(_doc(make_image_only_pdf()), StageMetric(name="t", latency_ms=0))
    assert out.method == "vlm" and "not installed" in out.notes[0]


def test_page_cap_limits_cost():
    vlm = FakeVLM(GOOD_TEXT)
    out = engine(None, vlm, max_pages=2).run(_doc(make_image_only_pdf(5)), StageMetric(name="t", latency_ms=0))
    assert vlm.calls == 2 and any("first 2 of 5" in n for n in out.notes)


def test_everything_fails_is_a_clear_unreadable_error():
    e = engine(None, FakeVLM(fail=True))
    with pytest.raises(OcrFailed):
        e.run(_doc(make_image_only_pdf()), StageMetric(name="t", latency_ms=0))
    with pytest.raises(DocumentError) as exc:
        extract_text("scan.pdf", make_image_only_pdf(), ocr=e)
    assert exc.value.code == ErrorCode.NO_TEXT_LAYER and "OCR could not read it" in exc.value.message


def test_extract_text_reports_ocr_method_and_warning():
    doc = extract_text("scan.pdf", make_image_only_pdf(), ocr=engine(None, FakeVLM(GOOD_TEXT)))
    assert doc.method == "ocr-vlm" and doc.metric.detail == "ocr-vlm" and "scanned image" in doc.warnings[0]


# ---------------------------------------------------------------- real Tesseract (skipped if not installed)

def test_real_tesseract_reads_scanned_resume():
    e = OcrEngine(OcrConfig(vlm_fallback=False))
    if e.tesseract is None:
        pytest.skip("Tesseract binary not available")
    doc = extract_text("scan.pdf", scanned_image_pdf(RESUME_A), ocr=e)
    assert doc.method == "ocr-tesseract"
    assert "Razorfin Payments" in doc.text and "Kafka-based event pipeline" in doc.text
