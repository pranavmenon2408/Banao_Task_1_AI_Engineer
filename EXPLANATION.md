# Explanation: Resume ↔ JD Fit Scorer

**Setup:** Agent 1 turns the resume into a structured profile. Agent 2 pulls criteria out of the JD and gives each
a level from 0 to 4 with quoted evidence. Plain Python then checks the quotes against the resume and computes the
score from `config/scoring.yaml`. The model is Qwen2.5-72B-Instruct via Hugging Face. Numbers are from
`scripts/calibrate.py` (3 sample resumes × 3 runs; raw results in `reports/` and `docs/DEVLOG.md`).

### 1. Design parameter: scorer temperature (kept at 0, one sample)

I first wanted to give the scorer some temperature (0.4–0.5) so it wouldn't be too rigid. Instead of guessing, I
scored the same three resumes 3 times each with each setting:

| Scorer setting | Worst score range for one resume | Tokens per resume |
|---|---|---|
| T = 0, 1 sample | 3.8 | ~3.7k |
| T = 0.5, 1 sample | **13.2** (strong resume dropped to 80.7 once) | ~3.5k |
| T = 0.5, median of 3 samples | 3.7 | **~8.1k** |

T = 0.5 alone made the same resume jump around, which is exactly what the task says not to do. The median of 3
fixed that but only matched T = 0 at over twice the tokens. So I kept T = 0 and put the effort into a stricter
rubric and scoring the JD section by section (experience, skills, education, domain) in parallel calls.

### 2. A failure I saw: the scorer quoted the JD as if it were the resume

On my first full run the strong candidate got level 2 for "Kubernetes familiarity", with the evidence
*"Familiarity with Kubernetes."* That sentence is from the **job description**; the resume never mentions
Kubernetes. The criteria and the profile were in the same prompt in the same format (JSON strings), so the model
mixed up which one it was quoting. It only got caught because I'd written a check that searches for every quote
in the actual resume text. It scored 0.43 against my 0.80 threshold, so it was flagged and lost a level. I then
put the criteria and profile in clearly separated blocks, switched the profile to Markdown and added a "never quote
the criteria" rule. I kept the code check, because the prompt alone can't be trusted.

### 3. Metric: how much one resume's score moves between identical runs

The first version moved **16 points** at temperature 0 (92.3 → 76.3). Looking at per-criterion levels, four
criteria flipped between 3 ("meets") and 4 ("exceeds") *together*. The model was forming an overall impression
instead of judging each criterion, and each flip cost 25 points. Making level 4 require a specific written condition
and changing the points to 0/25/55/85/100 brought it to 3.8. Sectioned scoring brought it to **3.1**. The two similar
resumes now land **2.5 points apart** (93.8 vs 91.3) and the weak one sits at 30. I also logged latency and tokens per
stage: sectioned is faster (~40 s vs ~46 s) but uses about 2× the tokens, which is still around $0.003 per resume.

### 4. What I didn't finish and what I'd do next

- **Real accuracy:** I showed consistency, not correctness. Three resumes I wrote for one JD isn't a test set. Next
  I'd get 30–50 real pairs labelled by a recruiter and check how often I'm within one level of them.
- **OCR:** scanned PDFs are detected and rejected with a clear message, but not read. I'd add Tesseract as a fallback.
- **Token cost:** the full rubric is repeated in every section call. I'd trim each call down to its own rules.
- **Docker:** written but not built, because Docker isn't installed on my laptop. Everything was tested locally.
- **Multi-column PDFs** can extract in the wrong reading order, and I haven't handled that.
