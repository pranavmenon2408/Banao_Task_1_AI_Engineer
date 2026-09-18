"""OCR for scanned (image-only) PDFs: Tesseract first, a vision-language model as fallback.

The order follows cost and latency. Tesseract runs locally, costs nothing and keeps the document on the server
(roughly 1 s per page at 300 dpi). The vision model is a network call that is several times slower and billed per
image token, so it only runs when Tesseract is missing, fails, or reports low confidence (poor scans, photos,
unusual fonts). Both are capped at `max_pages`, and vision-model pages are transcribed in parallel.

A vision model may silently "correct" or invent text, which quote verification would then treat as genuine. The
transcription prompt forbids this, and callers flag every OCR'd document to the user.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from types import ModuleType

import pymupdf
from PIL import Image

from app.agents.prompts import VLM_TRANSCRIBE_PROMPT
from app.core.config import OcrConfig, Settings, get_settings
from app.core.schemas import StageMetric
from app.llm.client import LLMClient, LLMError

log = logging.getLogger(__name__)


class OcrFailed(Exception):
    """Neither OCR engine produced usable text. `notes` explains what each engine did."""

    def __init__(self, notes: list[str]) -> None:
        super().__init__("; ".join(notes))
        self.notes = notes


@dataclass
class OcrOutcome:
    """Text recovered by OCR and which engine produced it."""

    text: str
    method: str  # "tesseract" | "vlm"
    pages: int
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)


def _render(page: pymupdf.Page, dpi: int) -> Image.Image:
    """Rasterise a page to a greyscale image."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return Image.open(io.BytesIO(pix.tobytes("png")))


class OcrEngine:
    """Runs Tesseract, then the vision model if needed, on the pages of a PDF."""

    def __init__(self, cfg: OcrConfig, settings: Settings | None = None, vlm: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.s = settings or get_settings()
        self._vlm = vlm

    @cached_property
    def tesseract(self) -> ModuleType | None:
        """The pytesseract module if the Tesseract binary is usable, else None."""
        try:
            import pytesseract

            if self.s.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.s.tesseract_cmd
            pytesseract.get_tesseract_version()
            return pytesseract
        except Exception as exc:  # binary missing or not on PATH
            log.warning("Tesseract unavailable: %s", exc)
            return None

    @property
    def vlm(self) -> LLMClient:
        """The vision-model client, created on first use."""
        if self._vlm is None:
            self._vlm = LLMClient(self.s, role="vlm")
        return self._vlm

    # ------------------------------------------------------------------ engines

    def _tesseract_page(self, page: pymupdf.Page) -> tuple[str, float, int]:
        """OCR one page in a single pass; returns (text, confidence x chars, chars) for weighted averaging."""
        pt = self.tesseract
        data = pt.image_to_data(_render(page, self.cfg.tesseract_dpi), output_type=pt.Output.DICT)
        lines: dict[tuple, list[str]] = {}
        conf_sum, n_chars = 0.0, 0
        for i, word in enumerate(data["text"]):
            word = word.strip()
            conf = float(data["conf"][i])
            if not word or conf < 0:
                continue
            lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(word)
            conf_sum += conf * len(word)  # weight by length so stray 1-char noise doesn't dominate
            n_chars += len(word)
        text = "\n".join(" ".join(ws) for ws in lines.values())
        return text, conf_sum, n_chars

    def _vlm_page(self, page: pymupdf.Page, metric: StageMetric) -> str:
        """Transcribe one page with the vision model; empty string if it reports no text."""
        # Image tokens scale with pixels; ~1600 px on the long side keeps 9-10 pt text legible.
        dpi = int(min(150, self.cfg.vlm_max_side_px / (max(page.rect.width, page.rect.height) / 72)))
        buf = io.BytesIO()
        _render(page, dpi).save(buf, format="JPEG", quality=85)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": VLM_TRANSCRIBE_PROMPT},
                ],
            }
        ]
        text = self.vlm.complete_text(messages, metric, max_tokens=2500).strip()
        return "" if text.upper().startswith("NO_TEXT") else text

    # ------------------------------------------------------------------ orchestration

    def run(self, doc: pymupdf.Document, metric: StageMetric) -> OcrOutcome:
        """OCR the document, recording vision-model usage in `metric`.

        Raises:
            OcrFailed: neither engine produced enough text.
        """
        pages = list(doc)[: self.cfg.max_pages]
        notes: list[str] = []
        if doc.page_count > len(pages):
            notes.append(f"Only the first {len(pages)} of {doc.page_count} pages were OCR'd.")

        if self.tesseract is None:
            notes.append("Tesseract is not installed")
        else:
            t = time.perf_counter()
            try:
                results = [self._tesseract_page(p) for p in pages]
                text = "\n\n".join(r[0] for r in results)
                chars = sum(r[2] for r in results)
                conf = sum(r[1] for r in results) / chars if chars else 0.0
                log.info("Tesseract: %d pages, %.1f s, confidence %.1f", len(pages), time.perf_counter() - t, conf)
                if chars >= self.cfg.min_chars and conf >= self.cfg.tesseract_min_confidence:
                    return OcrOutcome(text, "tesseract", len(pages), round(conf, 1), notes)
                notes.append(
                    f"Tesseract read {chars} chars at confidence {conf:.0f} "
                    f"(needs >= {self.cfg.min_chars} chars and >= {self.cfg.tesseract_min_confidence:.0f})"
                )
            except Exception as exc:
                notes.append(f"Tesseract failed: {type(exc).__name__}: {exc}")

        if not self.cfg.vlm_fallback:
            raise OcrFailed(notes + ["vision-model fallback is disabled"])
        t = time.perf_counter()
        parts: list[str] = [""] * len(pages)
        metrics = [StageMetric(name="vlm_page", latency_ms=0) for _ in pages]
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, min(self.cfg.vlm_max_parallel, len(pages)))) as pool:
            futures = {
                pool.submit(self._vlm_page, p, m): i for i, (p, m) in enumerate(zip(pages, metrics, strict=True))
            }
            for f, i in futures.items():
                try:
                    parts[i] = f.result()
                except LLMError as exc:
                    errors.append(f"page {i + 1}: {exc.message[:120]}")
        for m in metrics:
            metric.llm_calls += m.llm_calls
            metric.prompt_tokens += m.prompt_tokens
            metric.completion_tokens += m.completion_tokens
            metric.retries += m.retries
        text = "\n\n".join(p for p in parts if p)
        log.info("VLM OCR: %d pages, %.1f s, %d chars", len(pages), time.perf_counter() - t, len(text))
        if errors:
            notes.append("Vision model failed on " + "; ".join(errors))
        if len(text) >= self.cfg.min_chars:
            return OcrOutcome(text, "vlm", len(pages), None, notes)
        notes.append(f"Vision model returned {len(text)} chars")
        raise OcrFailed(notes)
