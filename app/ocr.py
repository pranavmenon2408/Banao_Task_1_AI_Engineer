"""OCR for scanned (image-only) PDFs: Tesseract first, a Hugging Face vision model as backup.

Order is decided by cost and latency, measured on samples/resume_a_scanned.pdf:
  Tesseract @300 dpi   ~0.9 s/page, local, free, no data leaves the server, 100% of lines exact
  Llama-4-Scout (VLM)  ~6 s/page, ~2.7k tokens/page over the network,  100% of lines exact
So the VLM only runs when Tesseract is missing, errors, or reports low confidence (smudged scans,
photos, handwriting-style fonts). Both are capped at `max_pages` because resumes rarely exceed 3-4 pages
and every extra page is pure latency/cost. VLM pages run in parallel.

A vision model can "tidy up" or invent text, which the grounding check would then treat as real.
The prompt forbids that, and the API warns the recruiter whenever a resume was read by OCR.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property

import pymupdf
from PIL import Image

from app.config import OcrConfig, Settings, get_settings
from app.llm import LLMClient, LLMError
from app.prompts import VLM_TRANSCRIBE_PROMPT
from app.schemas import StageMetric

log = logging.getLogger(__name__)


class OcrFailed(Exception):
    def __init__(self, notes: list[str]):
        super().__init__("; ".join(notes))
        self.notes = notes


@dataclass
class OcrOutcome:
    text: str
    method: str                  # "tesseract" | "vlm"
    pages: int
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)


def _render(page: pymupdf.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return Image.open(io.BytesIO(pix.tobytes("png")))


class OcrEngine:
    def __init__(self, cfg: OcrConfig, settings: Settings | None = None, vlm: LLMClient | None = None):
        self.cfg = cfg
        self.s = settings or get_settings()
        self._vlm = vlm

    @cached_property
    def tesseract(self):
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
        if self._vlm is None:
            self._vlm = LLMClient(model=self.s.hf_vlm_model, provider=self.s.hf_vlm_provider)
        return self._vlm

    # ------------------------------------------------------------------ engines

    def _tesseract_page(self, page: pymupdf.Page) -> tuple[str, float, int]:
        """One Tesseract pass returning text and word confidences (image_to_data), not two passes."""
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
            conf_sum += conf * len(word)   # weight by length so stray 1-char noise doesn't dominate
            n_chars += len(word)
        text = "\n".join(" ".join(ws) for ws in lines.values())
        return text, conf_sum, n_chars

    def _vlm_page(self, page: pymupdf.Page, metric: StageMetric) -> str:
        # Cap the long side: image tokens scale with pixels, and ~1600 px keeps 9-10 pt text legible.
        dpi = int(min(150, self.cfg.vlm_max_side_px / (max(page.rect.width, page.rect.height) / 72)))
        buf = io.BytesIO()
        _render(page, dpi).save(buf, format="JPEG", quality=85)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                 {"type": "text", "text": VLM_TRANSCRIBE_PROMPT}]}]
        text = self.vlm.complete_text(messages, metric, max_tokens=2500).strip()
        return "" if text.upper().startswith("NO_TEXT") else text

    # ------------------------------------------------------------------ orchestration

    def run(self, doc: pymupdf.Document, metric: StageMetric) -> OcrOutcome:
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
                notes.append(f"Tesseract read {chars} chars at confidence {conf:.0f} "
                             f"(needs >= {self.cfg.min_chars} chars and >= {self.cfg.tesseract_min_confidence:.0f})")
            except Exception as exc:
                notes.append(f"Tesseract failed: {type(exc).__name__}: {exc}")

        if not self.cfg.vlm_fallback:
            raise OcrFailed(notes + ["vision-model fallback is disabled"])
        t = time.perf_counter()
        parts: list[str] = [""] * len(pages)
        metrics = [StageMetric(name="vlm_page", latency_ms=0) for _ in pages]
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, min(self.cfg.vlm_max_parallel, len(pages)))) as pool:
            futures = {pool.submit(self._vlm_page, p, m): i for i, (p, m) in enumerate(zip(pages, metrics))}
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
