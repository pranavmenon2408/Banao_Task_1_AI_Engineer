"""Generate PDF/DOCX versions of the sample files plus deliberately unreadable resumes.

    python -m scripts.make_samples

Outputs in samples/:
  resume_a_strong.pdf              text PDF of resume A
  jd_backend_engineer.docx         the JD as a Word document
  unreadable_scanned_resume.pdf    image-only PDF (what a phone scan looks like to a parser)
  unreadable_encrypted_resume.pdf  password-protected PDF
  unreadable_corrupt_resume.pdf    truncated PDF bytes
"""

from __future__ import annotations

import io
import sys
import textwrap
from pathlib import Path

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "samples"


def text_pdf(lines: list[str], encrypt: str | None = None) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, encrypt=encrypt)
    y = 800
    for raw in lines:
        for line in textwrap.wrap(raw, 95) or [""]:
            c.setFont("Helvetica-Bold" if raw.isupper() and raw.strip() else "Helvetica", 10)
            c.drawString(50, y, line)
            y -= 14
            if y < 50:
                c.showPage()
                y = 800
    c.save()
    return buf.getvalue()


def scanned_pdf(lines: list[str]) -> bytes:
    """Render text as tiny filled rectangles: visually a page of text, but no text layer."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFillGray(0.15)
    y = 800
    for line in lines:
        x = 50
        for word in line.split():
            w = 5.2 * len(word)
            c.rect(x, y, w, 7, fill=1, stroke=0)
            x += w + 4
        y -= 14
    c.showPage()
    c.save()
    return buf.getvalue()


def two_column_pdf(left: list[str], right: list[str]) -> bytes:
    """Sidebar + main column, drawn row by row across both columns, like many resume builders do.
    A reader that follows content-stream order interleaves the two columns line by line."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 805, "ANANYA RAO  -  Senior Software Engineer | Backend & Distributed Systems")
    left_w = [ln for raw in left for ln in (textwrap.wrap(raw, 30) or [""])]
    right_w = [ln for raw in right for ln in (textwrap.wrap(raw, 62) or [""])]
    y = 775
    for i in range(max(len(left_w), len(right_w))):
        for x, lines in ((40, left_w), (215, right_w)):
            if i < len(lines):
                c.setFont("Helvetica-Bold" if lines[i].isupper() and lines[i].strip() else "Helvetica", 9)
                c.drawString(x, y, lines[i])
        y -= 13
    c.save()
    return buf.getvalue()


def scanned_image_pdf(lines: list[str]) -> bytes:
    """A realistic scan: text rendered to a bitmap at 150 dpi, slightly rotated, with speckle noise.
    There is no text layer, so only OCR can read it."""
    import random

    import pymupdf as fitz
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    w, h = 1240, 1754  # A4 at 150 dpi
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=19)
    bold = ImageFont.load_default(size=22)
    y = 70
    for raw in lines:
        for line in textwrap.wrap(raw, 100) or [""]:
            d.text((70, y), line, fill=20, font=bold if raw.isupper() and raw.strip() else font)
            y += 30
    rnd = random.Random(7)
    for _ in range(4000):
        img.putpixel((rnd.randrange(w), rnd.randrange(h)), rnd.randrange(120, 200))
    img = img.rotate(0.8, fillcolor=255, resample=Image.BICUBIC).filter(ImageFilter.GaussianBlur(0.6))
    png = io.BytesIO()
    img.save(png, format="PNG")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=png.getvalue())
    return doc.tobytes()


def main() -> None:
    a = (S / "resume_a_strong.txt").read_text(encoding="utf-8").splitlines()
    (S / "resume_a_strong.pdf").write_bytes(text_pdf(a))
    (S / "resume_a_scanned.pdf").write_bytes(scanned_image_pdf(a))

    # Same content as resume A, laid out as sidebar (skills/education/contact) + main column (experience)
    txt = "\n".join(a)
    body, rest = txt.split("EDUCATION", 1)
    edu, skills = rest.split("SKILLS", 1)
    left = (
        ["CONTACT", "Bengaluru, India", "ananya.rao@example.com", "+91 98450 00000", "", "SKILLS"]
        + [s.strip() for s in skills.strip().split(",")]
        + ["", "EDUCATION"]
        + edu.strip().splitlines()
    )
    right = [ln for ln in body.splitlines()[3:]]
    (S / "resume_a_two_column.pdf").write_bytes(two_column_pdf(left, right))
    (S / "unreadable_scanned_resume.pdf").write_bytes(scanned_pdf(a))
    (S / "unreadable_encrypted_resume.pdf").write_bytes(text_pdf(a, encrypt="candidate123"))
    (S / "unreadable_corrupt_resume.pdf").write_bytes(text_pdf(a)[:400])

    d = Document()
    for line in (S / "jd_backend_engineer.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("- "):
            d.add_paragraph(line[2:], style="List Bullet")
        elif line.strip():
            d.add_paragraph(line)
    d.save(S / "jd_backend_engineer.docx")
    print("samples written to", S)


if __name__ == "__main__":
    sys.exit(main())
