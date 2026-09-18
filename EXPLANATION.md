# Explanation: Resume ↔ JD Fit Scorer

**Setup:** Agent 1 turns the resume into a structured profile. Agent 2 pulls criteria from the JD and gives each a
level from 0 to 4, quoting the resume. Plain Python checks every quote is really in the resume and computes the score
from `config/scoring.yaml`. It runs on Llama-3.3-70B via Hugging Face. I tuned it on Qwen2.5-72B, which went offline
near the end (point 4). The numbers are in `docs/DEVLOG.md`.

### 1. Design parameters: chunking, and scorer temperature

**Chunking:** I split the resume at its section headings, up to about 3,000 tokens per chunk, never mid-section. A
normal 1–3 page resume is 600–2,000 tokens, so it reaches Agent 1 in one piece and nothing gets lost in merging.
Fixed 512-token windows would cut a job title off from its bullets, and the model would give achievements to the wrong
job. Agent 2 needs no chunking because it only sees the short profile.

**Scorer temperature (kept at 0):** I wanted 0.4–0.5 at first, so I tested it on the same resumes, 3 runs each:

| Scorer setting | Biggest score swing for one resume | Tokens per resume |
|---|---|---|
| T = 0 | 3.8 | ~3.7k |
| T = 0.5 | **13.2** (strong resume dropped to 80.7 once) | ~3.5k |
| T = 0.5, median of 3 runs | 3.7 | **~8.1k** |

0.5 made the same resume jump around. The median fixed that but cost twice the tokens for no gain over T = 0.

### 2. Failures I actually saw

**The scorer quoted the JD as if it were the resume.** The strong candidate got level 2 for Kubernetes with the
evidence *"Familiarity with Kubernetes."*, which is a line from the JD. The resume never mentions Kubernetes. Both
were in one prompt in the same JSON format, so the model mixed them up. My quote check caught it (0.43 similarity,
needs 0.80). I split the prompt into clearly labelled blocks and kept the check.

**Two-column and scanned PDFs.** I made a two-column copy and a scanned copy of the same resume. pypdf read the
columns across the page, so sidebar skills landed inside experience bullets ("…transaction status / Kafka /
updates…"), and bullets only matched the resume at 0.93–0.96 instead of 1.00. The scan was rejected as unreadable. I
switched to PyMuPDF, reading by column position, and added OCR: Tesseract first (1.5 s, free), then a Hugging Face
vision model only if Tesseract is missing or unsure (~6 s, ~2.8k tokens per page). All three copies now get the same score.

### 3. Metric: how much the score changes when I run the same resume again

The same resume should get the same score every time. My first version gave one resume **92.3, then 76.3**. Four
criteria flipped between "meets" (3) and "exceeds" (4) together, each worth 25 points. Once "exceeds" needed a
specific written reason and the points became 0/25/55/85/100, the change dropped to about 4 points. Scoring the JD
section by section in separate calls was **not** more consistent in a fair side-by-side test, but it was about 30%
faster (22 s vs 31 s) for twice the tokens (still about $0.002 per resume). On Llama-3.3-70B all three resumes now get
the exact same score on every repeat, and the two similar ones are 0.7 apart.

### 4. What I didn't finish and what I'd do next

- **Real accuracy:** I showed consistency, not correctness. Next I'd get 30–50 real pairs scored by a recruiter and
  check how often I'm within one level of them.
- **Provider dependence:** Qwen's only fast host stopped serving it, and my API told the user "temporary, retry
  later", which was wrong. The switch to Llama also showed that my section split hid the skills list from a criterion
  the new model filed under "experience" (fixed). Next: a backup model list and a proper error message.
- **Trusting OCR text:** a vision model can "clean up" or invent words that my quote check would accept. For now I only
  warn the recruiter. Tesseract is also set up for English only.
- **Harder layouts and cost:** tables and three columns are untested, and the full rubric repeats in every section call.
- **Docker:** written but never built, because Docker isn't installed on my laptop. Everything was tested locally.
