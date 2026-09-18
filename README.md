# Resume ↔ Job Description Fit Scorer

Give it a job description and a resume. It returns a score out of 100 that a recruiter can act on, broken down
per requirement. Every requirement shows the line from the JD it came from, the resume text that supports it
(checked against the actual resume), a short reason and what's missing.

**How it works (two agents):**

1. **Parse** the files (PDF / DOCX / TXT). PDFs are read column by column, so two-column resumes come out in the
   right order. Scanned resumes are OCR'd with Tesseract, with a Hugging Face vision model as backup. Files that
   still can't be read (encrypted, corrupt, blank scans) are rejected with a specific error and a hint.
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

By default it runs on Hugging Face, which needs an access token with the **"Make calls to Inference Providers"**
permission (https://huggingface.co/settings/tokens). You can use another provider instead; see
[Choosing a provider and model](#choosing-a-provider-and-model).

```bash
git clone https://github.com/pranavmenon2408/Banao_Task_1_AI_Engineer.git
cd Banao_Task_1_AI_Engineer
cp .env.example .env          # then put your key in HF_TOKEN (or your provider's key variable)
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

**Optional, for scanned resumes:** install [Tesseract](https://github.com/tesseract-ocr/tesseract)
(Windows: the UB-Mannheim installer; macOS: `brew install tesseract`; Ubuntu: `apt install tesseract-ocr`). If it
isn't on your PATH, set `TESSERACT_CMD` in `.env`. Without Tesseract, scanned resumes go straight to the vision
model, which is slower (~6 s/page) and uses tokens. The Docker image already includes Tesseract.

Start the API and the UI in two terminals:

```bash
uvicorn app.api.main:app --port 8000
streamlit run ui/streamlit_app.py
```

Then open http://localhost:8501.

---

## Usage

### In the UI

1. **Job description:** upload a PDF / DOCX / TXT, or paste the text.
2. **Resume:** upload a PDF / DOCX / TXT (upload only).
3. Click **Assess fit**. A run takes roughly 10–25 s with the default model.
4. Use the sidebar sliders to change the weights. The score updates instantly without calling the LLM again.

Try it with the files in `samples/`. `resume_a_two_column.pdf` and `resume_a_scanned.pdf` are the same resume as
`resume_a_strong.pdf` in harder formats (all three score the same). `unreadable_*.pdf` shows how bad files are handled.

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

### Choosing a provider and model

The text model (used by both agents) and the OCR vision model are each set by a **provider**, a **model** and
optional **fallback models** in `.env`. Any provider with an OpenAI-compatible chat API works.

| `LLM_PROVIDER` | Key variable | Example `LLM_MODEL` |
|---|---|---|
| `huggingface` (default) | `HF_TOKEN` | `meta-llama/Llama-3.3-70B-Instruct` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `google` (Gemini) | `GOOGLE_API_KEY` | `gemini-2.0-flash` |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `mistral` | `MISTRAL_API_KEY` | `mistral-large-latest` |
| `openrouter` | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct` |
| `together` | `TOGETHER_API_KEY` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `ollama` (local) | none | `llama3.1:8b` |
| `openai-compatible` | `LLM_API_KEY` (if needed) | set `LLM_ENDPOINT` to the `/chat/completions` URL |

(Model names are examples; check each provider's current list.) For example, to use OpenAI:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

- **`LLM_FALLBACK_MODELS`** (comma-separated) are tried in order when the main model isn't being served.
  The result then carries a warning, because a different model can score differently.
- **`HF_INFERENCE_PROVIDER`** (Hugging Face only) picks which host serves the model. `auto` uses the first provider
  enabled on your account.
- **`VLM_PROVIDER` / `VLM_MODEL` / `VLM_FALLBACK_MODELS`** do the same for the OCR vision model, which must accept
  images. `VLM_PROVIDER` defaults to `LLM_PROVIDER`.
- If a model isn't available, the API returns `503 MODEL_NOT_AVAILABLE` with a hint to change the config. It doesn't
  suggest retrying, because retrying won't help. `GET /health` shows the provider, model and fallbacks in use.
- The calibration numbers were measured on Hugging Face models. After switching, re-check with
  `python -m scripts.calibrate --repeats 3`.

The provider layer is `app/providers.py` (registry + transports) and `app/llm.py` (`LLMClient`: retries,
fallback step-down, JSON repair).

### Configuration

- **`.env`:** provider, model and fallbacks for the text and vision models (see above), `TESSERACT_CMD`, timeouts,
  retries. The old `HF_MODEL` / `HF_PROVIDER` / `HF_VLM_MODEL` names still work.
- **`config/scoring.yaml`:** importance and category weights, points per rubric level, knockout rule,
  recommendation bands, grounding threshold, scoring mode, scorer temperature and samples, OCR settings
  (page cap, Tesseract DPI and confidence threshold, vision-model fallback).

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
- Two-column PDFs are read in the right order (sidebar, then main column).
- Scanned resumes are OCR'd: Tesseract first (~1.5 s/page, free), then a vision model if Tesseract is missing or
  unsure (~6 s and ~2.8k tokens/page). The parse method and its cost show up in the run metrics.
- Resumes that really can't be read are detected and explained, not crashed on: blank or illegible scans,
  password-protected PDFs, corrupt files, garbled font encodings, wrong file types, near-empty files.
- LLM failures are handled by type: timeouts and rate limits are retried, a model that isn't being served steps down
  to fallback models, bad JSON gets one repair turn, and anything left is reported as a clean error with the right hint.
- Any LLM provider: Hugging Face, OpenAI, Gemini, Groq, Mistral, OpenRouter, Together, Ollama or a custom endpoint.

**Doesn't (yet)**
- OCR'd text is trusted as-is. A vision model could "clean up" or invent words, and the recruiter only gets a warning.
  Tesseract is set up for English only.
- Calibration is only checked on **3 hand-written samples for one JD**, which isn't a real accuracy evaluation.
- Only Hugging Face was tested live. The other providers go through the same OpenAI-compatible transport, which is
  covered by mocked HTTP tests, not real API calls.
- The calibration history in `docs/DEVLOG.md` was measured on Qwen2.5-72B, then re-checked on Llama-3.3-70B.
- No authentication, no database (results live in a local JSON cache and `data/runs.jsonl`).
- The Docker setup was written but not built on my machine (Docker isn't installed there). The local setup was tested.
- Only two-column layouts were tested. Tables, three columns and floating text boxes may still come out in the wrong order.
- An unverifiable quote only costs one level (`grounding.ungrounded_level_penalty`), so a hallucinated "meets" (3)
  becomes "partial" (2), not 0. That is deliberately lenient because PDF extraction sometimes mangles real quotes.
