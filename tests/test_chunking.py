from app.chunking import chunk_resume, estimate_tokens, split_sections


def _long_resume(jobs: int = 30) -> str:
    parts = ["John Smith\nSUMMARY\nEngineer with broad experience.", "EXPERIENCE"]
    for i in range(jobs):
        bullets = "\n".join(f"- Delivered project {i}.{b} improving throughput and reliability for users" for b in range(8))
        parts.append(f"Engineer {i}, Company {i} (2000 - 2001)\n{bullets}\n")
    parts += ["EDUCATION\nB.Sc Physics, Some University, 1999", "SKILLS\nPython, Go, SQL"]
    return "\n".join(parts)


def test_short_resume_is_single_chunk():
    text = "SUMMARY\nShort resume.\nSKILLS\nPython"
    assert chunk_resume(text, 3000) == [text]


def test_sections_detected():
    heads = [s.heading for s in split_sections(_long_resume(2))]
    assert heads[:3] == ["SUMMARY", "EXPERIENCE", "EDUCATION"] or "EXPERIENCE" in heads


def test_long_resume_chunks_respect_budget_and_keep_all_content():
    text = _long_resume(30)
    chunks = chunk_resume(text, budget_tokens=800)
    assert len(chunks) > 1
    assert all(estimate_tokens(c) <= 800 + 20 for c in chunks)
    joined = "\n".join(chunks)
    for i in range(30):
        assert f"Company {i} " in joined
    assert "EDUCATION" in joined and "B.Sc Physics" in joined


def test_oversized_section_repeats_heading():
    chunks = chunk_resume(_long_resume(30), budget_tokens=800)
    assert sum(c.startswith("EXPERIENCE") for c in chunks) >= 2
    # A role's title line and its bullets should stay together (paragraph-level split)
    for c in chunks:
        if "Engineer 5, Company 5" in c:
            assert "project 5.7" in c
