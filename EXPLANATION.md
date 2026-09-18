# Explanation: Resume ↔ JD Fit Scorer

**Setup:** Agent 1 turns the resume into a structured profile. Agent 2 pulls criteria from the JD and gives each a
level from 0 to 4, quoting the resume. Plain Python checks every quote is really in the resume and computes the score
from `config/scoring.yaml`. The model is Qwen2.5-72B via Hugging Face. The numbers are in `docs/DEVLOG.md`.

### 1. Design parameters: chunking, and scorer temperature

**Chunking:** I split the resume at its section headings, up to about 3,000 tokens per chunk, never mid-section. A
normal 1–3 page resume is 600–2,000 tokens, so it reaches Agent 1 in one piece and nothing gets lost in merging (all
my samples were one chunk). I avoided fixed 512-token windows because they cut a job title off from its bullets, and
the model then gives achievements to the wrong job. Agent 2 needs no chunking because it only sees the short profile.

**Scorer temperature (kept at 0):** I wanted 0.4–0.5 at first, so I tested it on the same resumes, 3 runs each:

| Scorer setting | Biggest score swing for one resume | Tokens per resume |
|---|---|---|
| T = 0 | 3.8 | ~3.7k |
| T = 0.5 | **13.2** (strong resume dropped to 80.7 once) | ~3.5k |
| T = 0.5, median of 3 runs | 3.7 | **~8.1k** |

0.5 made the same resume jump around. The median fixed that but cost twice the tokens for no gain over T = 0.

### 2. Failures I actually saw

**The scorer quoted the JD as if it were the resume.** The strong candidate got level 2 for Kubernetes with the
evidence *"Familiarity with Kubernetes."*, which is a line from the JD. The resume never mentions Kubernetes. Criteria
and profile were in one prompt in the same JSON format, so the model mixed them up. My quote check caught it (0.43
similarity, needs 0.80). I then split the prompt into clearly labelled blocks and kept the check.

**Two-column and scanned PDFs.** I made a two-column copy and a scanned-image copy of the same resume. pypdf read the
columns across the page, so sidebar skills landed inside experience bullets ("…transaction status / Kafka /
updates…"). Agent 1 mostly repaired it, but bullets only matched the resume at 0.93–0.96 instead of 1.00. The scan
was just rejected as unreadable. I switched to PyMuPDF and read by column position, and added OCR: Tesseract first
(1.5 s, free), then a Hugging Face vision model only if Tesseract is missing or unsure (~6 s, ~2.8k tokens per page).
All three copies now score 93.3.

### 3. Metric: how much the score changes when I run the same resume again

The same resume should get the same score every time. My first version gave one resume **92.3, then 76.3**. Four
criteria were flipping between "meets" (3) and "exceeds" (4) together, and each flip was worth 25 points. Once
"exceeds" needed a specific written reason and the points became 0/25/55/85/100, the change dropped to about 4 points,
and the two similar resumes now land within 1–5 points of each other. I also tried scoring the JD section by section
in separate calls. In a fair side-by-side test it was **not** more consistent, but it was about 30% faster (22 s vs
31 s) and followed the section rules better, for twice the tokens (still about $0.002 per resume).

### 4. What I didn't finish and what I'd do next

- **Real accuracy:** I showed consistency, not correctness. Next I'd get 30–50 real pairs scored by a recruiter and
  check how often I'm within one level of them.
- **Trusting OCR text:** a vision model can "clean up" or invent words that my quote check would then accept. For now
  I only warn the recruiter. Tesseract is also set up for English only.
- **Harder layouts:** tables, three columns and text boxes are untested; I only tested two columns.
- **Cost:** the full rubric is repeated in every section call; I'd trim each call to its own rules.
- **Docker:** written but never built, because Docker isn't installed on my laptop. Everything was tested locally.
