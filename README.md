# Resume ↔ Job Description Fit Scorer

Give it a job description and a resume. It returns a score out of 100 that a recruiter can act on, broken down
per requirement. Every requirement shows the line from the JD it came from, the resume text that supports it
(checked against the actual resume), a short reason and what's missing.

**How it works (two agents):**

1. **Parse** the files (PDF / DOCX / TXT). Unreadable files (scanned, encrypted, corrupt, garbled) are rejected
   with a specific error and a hint.
2. **Agent 1** turns the resume into a structured profile (roles, dated bullets copied verbatim, skills, education).
   Long resumes are split at section headings and merged back together.
3. **Agent 2** pulls the hiring criteria out of the JD (cached per JD, so every resume is judged against the same
   list). It then gives each criterion a 0–4 level using a fixed rubric, quoting resume evidence.
4. **Code, not the LLM:** checks every quote actually exists in the resume, computes years of experience from the
   dates, and turns levels into the final score using the weights in [`config/scoring.yaml`](config/scoring.yaml).

The design decisions and measurements are in **[EXPLANATION.md](EXPLANATION.md)**. The raw build notes are in
[docs/DEVLOG.md](docs/DEVLOG.md).

```
app/            FastAPI backend: parsing, chunking, agents, grounding, scoring
ui/             Streamlit front end (talks to the API over HTTP)
config/         scoring.yaml: weights, rubric points, thresholds, scorer settings
samples/        sample JD, three resumes for calibration, unreadable resume PDFs
scripts/        calibrate.py (consistency check), make_samples.py
tests/          pytest suite (no API key needed)
reports/        calibration results
```

---

## Setup

You need a Hugging Face access token with the **"Make calls to Inference Providers"** permission
(https://huggingface.co/settings/tokens).

```bash
git clone https://github.com/pranavmenon2408/Banao_Task_1_AI_Engineer.git
cd Banao_Task_1_AI_Engineer
cp .env.example .env          # then put your token in HF_TOKEN
```

### Option A: run with Docker

```bash
docker compose up --build
```

- UI: http://localhost:8501
- API docs (Swagger): http://localhost:8000/docs

`data/` (cache + run log) and `config/` are mounted, so you can change weights without rebuilding
(restart the `api` container to apply).

### Option B: run without Docker (Python 3.13)

```bash
# Windows
py -3.13 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.13 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Start the API and the UI in two terminals:

```bash
uvicorn app.main:app --port 8000
streamlit run ui/streamlit_app.py
```

Then open http://localhost:8501.

---

## Usage

### In the UI

1. **Job description:** upload a PDF / DOCX / TXT, or paste the text.
2. **Resume:** upload a PDF / DOCX / TXT (upload only).
3. Click **Assess fit**. A run takes roughly 30–60 s with the default model.
4. Use the sidebar sliders to change the weights. The score updates instantly without calling the LLM again.

Try it with the files in `samples/`. `samples/unreadable_*.pdf` shows how bad files are handled.

### Through the API

```bash
curl -F "resume=@samples/resume_a_strong.pdf" \
     -F "jd_file=@samples/jd_backend_engineer.docx" \
     http://localhost:8000/api/v1/score
```

| Endpoint | What it does |
|---|---|
| `POST /api/v1/score` | multipart: `resume` + (`jd_file` **or** `jd_text`), optional `weights` JSON. Returns the full assessment |
| `POST /api/v1/rescore` | re-weights an existing result, no LLM call |
| `GET /api/v1/config` | current weights and thresholds |
| `GET /health` | status + model |

Errors always come back as `{"error": {"code", "message", "hint", "field"}}`. For example, a scanned PDF gives
`422 NO_TEXT_LAYER`, and a model outage gives `503 LLM_UNAVAILABLE`.

### Configuration

- **`.env`:** token, model (`HF_MODEL`, default `Qwen/Qwen2.5-72B-Instruct`), provider, timeouts, retries.
- **`config/scoring.yaml`:** importance and category weights, points per rubric level, knockout rule,
  recommendation bands, grounding threshold, scoring mode, scorer temperature and samples.

### Tests and calibration

```bash
pip install -r requirements-dev.txt
pytest                                    # no token needed; the LLM is faked
python -m scripts.calibrate --repeats 3   # real LLM: scores 3 sample resumes 3x each
```

---

## What works and what doesn't

**Works**
- PDF / DOCX / TXT input for the JD, and PDF / DOCX / TXT upload for the resume.
- Per-criterion scores with reasoning, JD quote and verified resume evidence. Weights come from config and can be
  overridden per request.
- Consistency on the three calibration samples: the two similar resumes land a few points apart, not 40
  (numbers in [EXPLANATION.md](EXPLANATION.md) and `reports/`).
- Unreadable resumes are detected and explained, not crashed on: scanned or image-only PDFs, password-protected
  PDFs, corrupt files, garbled font encodings, wrong file types, near-empty files.
- LLM failures (timeouts, rate limits, bad JSON) are retried or repaired, then reported as clean errors.

**Doesn't (yet)**
- **No OCR.** Scanned resumes are rejected with advice instead of being read.
- Calibration is only checked on **3 hand-written samples for one JD**, which isn't a real accuracy evaluation.
- One run takes about 30–60 s, which depends on the Hugging Face provider's speed.
- No authentication, no database (results live in a local JSON cache and `data/runs.jsonl`).
- The Docker setup was written but not built on my machine (Docker isn't installed there). The local setup was tested.
- Multi-column PDF layouts can extract in the wrong reading order, which can hurt profile extraction.
- An unverifiable quote only costs one level (`grounding.ungrounded_level_penalty`), so a hallucinated "meets" (3)
  becomes "partial" (2), not 0. That is deliberately lenient because PDF extraction sometimes mangles real quotes.
