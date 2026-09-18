import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_text_pdf(lines: list[str], encrypt: str | None = None) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, encrypt=encrypt)
    y = 800
    for line in lines:
        c.drawString(50, y, line)
        y -= 16
        if y < 50:
            c.showPage()
            y = 800
    c.save()
    return buf.getvalue()


def make_image_only_pdf(pages: int = 1) -> bytes:
    """Stand-in for a scanned resume: vector shapes, no text layer."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for _ in range(pages):
        for i in range(40):
            c.rect(50, 780 - i * 18, 300 + (i * 37) % 200, 8, fill=1)
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def resume_lines() -> list[str]:
    return [
        "Jane Doe - Backend Engineer",
        "EXPERIENCE",
        "Senior Software Engineer, Acme Corp (2020 - Present)",
        "- Built REST APIs in Python and FastAPI serving 2M requests/day",
        "- Led migration of batch jobs to AWS Lambda, cutting cost by 30%",
        "EDUCATION",
        "B.Tech Computer Science, IIT Madras, 2019",
        "SKILLS",
        "Python, FastAPI, PostgreSQL, Docker, AWS, Redis",
    ]
