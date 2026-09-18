# Development log

Observations recorded while building, with the numbers that were actually measured. The one-page
explanation document draws on these.

## 1. First end-to-end run (Qwen/Qwen2.5-72B-Instruct, provider=auto, temperature 0)

Sample: `samples/resume_a_strong.txt` against `samples/jd_backend_engineer.txt`.

| Stage | Latency | Prompt tok | Completion tok |
|---|---|---|---|
| JD criteria extraction | 30.0 s | 760 | 748 |
| Resume profile extraction | 20.4 s | 777 | 565 |
| Criterion scoring | 36.5 s | 1744 | 891 |
| Grounding + aggregation | 4 ms | 0 | 0 |

Overall 72.5 ("Potential fit"); 9 criteria extracted.

**Observed failure: evidence quoted from the wrong document.** For the nice-to-have criterion
"Kubernetes Familiarity" the scorer returned level 2 with the evidence quote
`"Familiarity with Kubernetes."`. That phrase is from the *job description*; the resume never mentions
Kubernetes. The model had the criterion description in context and echoed it back as if it
were resume evidence. The grounding check measured 0.43 similarity against the resume text (threshold
0.80), marked the criterion ungrounded and dropped it one level (2 -> 1).
Cause: the scorer sees both criteria and profile in one prompt, and the criterion text is phrased like
a resume line. Fix applied: code-level verification (kept), plus the prompt now tells the scorer never to quote
the criteria (see later entries).

Also seen: a quote taken from the extractor's generated `summary` field ("Backend engineer with 6 years
of experience, speci...") matched at only 0.84, because the summary is written by Agent 1, not copied from the resume.
It passed the 0.80 threshold, but it is weaker evidence than a verbatim bullet.

## 2. Unhandled httpx timeout -> HTTP 500 (bug in my error handling)

Scoring `resume_a_strong.pdf` against the DOCX JD through the API returned **HTTP 500 after 122 s**, and the
calibration script crashed on its 7th run with the same traceback: `httpx.ReadTimeout: The read operation timed out`.
Cause: `huggingface_hub` 1.x sends requests through **httpx**, so a slow provider raises `httpx.ReadTimeout`. My retry
logic caught `InferenceTimeoutError`, `HfHubHTTPError` and the built-in `TimeoutError`/`ConnectionError`, but not httpx's
exception hierarchy, so the "transient, retry it" path never ran.
Fix: catch `httpx.TransportError` (covers timeouts, connect errors and protocol errors) as transient and retry with backoff; add a
catch-all handler so any unexpected exception still leaves the API as a structured `INTERNAL_ERROR` rather than a raw 500;
add a regression test (`tests/test_llm.py::test_httpx_timeout_is_retried`).

## 3. Calibration baseline: temperature 0 is not stable

`scripts/calibrate.py --repeats 3`, JSON scorer input (prompt v4), criteria cached, profile re-extracted every run.
(The run crashed on the 7th call because of the bug above, so B and C have 2 runs each.)

| Resume | Runs | Scores |
|---|---|---|
| A (strong) | 3 | 92.3, 91.3, **76.3** |
| B (strong, similar) | 2 | 94.8, **78.8** |
| C (adjacent) | 2 | 39.8, 39.8 |

- Mean A (86.6) vs mean B (86.8): similar resumes land close **on average**, but a *single* run of the same resume moved
  **16 points** at temperature 0 with a fixed seed. Provider-side inference is not deterministic.
- The flips are **correlated**: in the low runs, criteria c2-c5 (Python, DB, queues, cloud) all drop 4 -> 3 *together*.
  The scorer settles into either an "exceeds everything" mode or a "meets everything" mode, which is a halo effect from scoring all
  criteria in one call despite the "rate independently" instruction.
- The weak resume is perfectly stable; all instability sits on the **3-vs-4 boundary**, worth 25 points per criterion
  with the original `level_points` (75 / 100).

## 4. New rubric + points: drift gone; a grounding false positive I caused

After making level 4 require a named condition and moving `level_points` to 0/25/55/85/100
(markdown input, T=0, single call, 3 repeats):

| Resume | Baseline (old rubric) | New rubric |
|---|---|---|
| A | 92.3, 91.3, 76.3 | 90.1, 90.1, 90.1 |
| B | 94.8, 78.8 | 88.8, 88.8, 85.0 |
| C | 39.8, 39.8 | 23.2, 24.4, 24.4 |

|mean A - mean B| = 2.6. Worst single-resume range fell from 16.0 to 3.8 points.

**False positive in the grounding check (my bug).** A's Education was judged 3 by the scorer and then cut to 2 in
every run. The scorer had quoted the line my renderer builds by joining extracted fields,
"B.E. Computer Science, Computer Science, RV College of Engineering, 2018" (degree and field both contained "Computer
Science"). That string is not in the resume, so verification failed. Fixes: don't repeat the field when the degree
already contains it, and have the grounder accept a quote whose every comma/pipe-separated fragment is found (a single
invented fragment still fails). Genuine failures still fail: A's "Kubernetes" quotes score 0.43 / 0.0.

Also found while building sectioned scoring: the extractor reported `total_experience_years = 6.0` for A, while the role
dates (Jul 2018 - present, contiguous) give 8.2 years. It most likely copied "6 years" from the resume's own summary line. Years
are now computed in code from dates (overlaps merged, internships excluded) and shown to the scorer.

## 5. Scorer experiment matrix (same cached criteria, 3 resumes x 3 repeats each)

All with the new rubric and points. Reports are in `reports/`. Variants 3-5 ran after the fragment-grounding fix
(entry 4), so compare *raw* levels across all variants, or final scores within 3-5.

| # | Scorer variant | mean A / B / C | \|A-B\| | Range A / B / C | Tokens/run | Latency/run |
|---|---|---|---|---|---|---|
| 1 | single, markdown, T=0 | 90.1 / 87.5 / 24.0 | 2.6 | 0.0 / 3.8 / 1.2 | ~3.7k | ~46 s |
| 2 | single, JSON, T=0 | 93.8 / 93.3 / 30.1 | 0.5 | 1.5 / 9.0 / 5.7 | ~3.9k | ~53 s |
| 3 | single, markdown, T=0.5 | 89.3 / 86.3 / 25.9 | 3.0 | 13.2 / 2.3 / 4.6 | ~3.5k | ~76 s |
| 4 | single, markdown, T=0.5, median of 3 | 93.8 / 87.3 / 22.0 | 6.5 | 1.5 / 2.3 / 3.7 | ~8.1k | ~56 s |
| 5 | **sectioned, markdown, T=0** | 93.8 / 91.3 / 30.0 | 2.5 | 1.5 / 1.5 / 3.1 | ~7.3k | ~40 s |

- T=0.5 with one sample is the noisiest (A dropped to 80.7 once), as expected.
- Median-of-3 at T=0.5 gets back to T=0's stability but costs 2.2x the tokens, so it's not worth it here.
- JSON input: similar means but noisier (B range 9.0) and ~6% more tokens than markdown.
- Sectioned: the most stable (**wrong, confounded; see the fair re-test in entry 6**), and faster because 4 small calls run in parallel. Tokens roughly double because the rubric
  system prompt is repeated per section. At the provider's cost estimate for this model (~$0.37 / 1M tokens) that is
  about $0.003 per resume.
- Chosen default: **sectioned, markdown, T=0, 1 sample**. The latency outlier in run 3 (116-136 s) was provider-side
  slowness during that window, not the variant.

## 6. Correction: sectioned scoring is not more stable, only faster and more focused

Entry 5 compared sectioned vs single unfairly: the single run happened *before* the fragment-grounding fix
(its worst range, 3.8, came from a false "ungrounded" penalty) and without the section rules or code-computed years.
Re-run as a fair A/B: both modes use the same rules and computed years, Agent 1's profile is frozen (cached) so only
the scorer varies, both run **at the same time** (same provider load), 5 repeats each.

| | single call | sectioned |
|---|---|---|
| mean A / B / C | 93.9 / 88.8 / 28.7 | 93.3 / 93.9 / 29.9 |
| \|A - B\| | 5.1 | 0.6 |
| worst range (same resume, 5 runs) | 4.5 | 7.5 |
| criteria whose level changed across runs | 4 | 4 |
| latency per resume | ~31 s | ~22 s |
| tokens per resume | ~2.8k | ~6.1k |

Stability is a tie (sectioned is slightly worse on B). Sectioned is ~30% faster (4 small parallel calls) and applies
section rules more reliably: B's "deployed them to AWS EKS" got Kubernetes 2-3 in sectioned, but 1 in all 5 single
runs even though the single prompt contains the same "EKS = managed Kubernetes" rule, probably because the rule is buried in one long
prompt. Kept sectioned for speed and focus. The cost is 2.2x tokens (~$0.002 per resume at ~$0.37/1M).

## 7. Multi-column and scanned PDFs (observed, then fixed)

Built `samples/resume_a_two_column.pdf` (resume A's exact content as sidebar + main column, drawn row by row like
many resume builders) and `samples/resume_a_scanned.pdf` (A rendered as a slightly rotated, noisy 150-dpi image).

**Two-column, before (pypdf):** the columns were interleaved line by line, e.g. "...transaction status / Kafka /
updates, cutting reconciliation lag...". Agent 1 mostly repaired it (3 roles, all bullets correct; score 93.3), but
the raw text no longer contained the bullets verbatim: extracted highlights matched the resume text at 0.93-0.96
instead of 1.00, eating into the 0.80 grounding margin. Longer sidebar text would do more damage.
**Fix:** PyMuPDF with our own reading order. PyMuPDF merges text on the same baseline across columns into one line,
so lines are split at wide span gaps first; then a gutter must have >= 8% of page text and >= 5 lines per side, with
<= 15% of lines crossing it. A unit test caught my first version splitting a normal resume with right-aligned dates
into two columns (shares were measured against narrow lines only), which is why the crossing rule exists.

**Scanned, before:** rejected outright as `NO_TEXT_LAYER`, although it is perfectly legible.
**Fix:** OCR chain, with the order chosen by measured cost:

| Engine (1 page) | Time | Lines exact | Cost |
|---|---|---|---|
| Tesseract @100 / 150 / 200 dpi | 0.55-0.65 s | 94% | local, free |
| **Tesseract @300 dpi** | ~0.85 s (1.5 s incl. render in the API) | **100%** | local, free |
| Llama-4-Scout-17B (HF) | 6.0-6.3 s | 100% | ~2.8k tokens |
| Qwen2.5-VL-72B (HF) | 13.1 s | 100% | ~3.1k tokens |
| gemma-3-27b (HF) | 36.6 s | 94% | ~0.6k tokens |
| Qwen2.5-VL-7B | not served by any provider | | |

Tesseract runs first; the vision model only runs if Tesseract is missing, errors, or reports confidence < 70
(the clean scan scored ~94). Pages are capped at 4 and VLM pages run in parallel.

**After, same resume through the API:** text PDF 93.3, two-column PDF 93.3, scanned via Tesseract 93.3, scanned via
the VLM (Tesseract disabled) 93.3. The blank "scan" (rectangles, no letters) now fails in 2.4 s with the reason
("Tesseract read 0 chars ...; Vision model returned 0 chars").

## 8. Provider outage -> switched to Llama-3.3-70B, which exposed a routing bug

**Outage.** A scoring request from the UI failed in 382 ms with `model_not_supported`: "Qwen/Qwen2.5-72B-Instruct is not
supported by any provider you have enabled". The same API process had scored fine two hours earlier, so nothing local
had changed. Probing providers directly: novita (which `auto` had been using) was failing Hugging Face's own health
check and timed out; featherless-ai served the model but isn't in the account's enabled list; others don't host it.
Pinning `HF_PROVIDER=featherless-ai` "worked" but was unusable: sectioned scoring's 4 parallel calls got
**429 Too Many Requests**, successful calls took ~90 s, and two returned invalid JSON (repair turns). One request ran
past 5 minutes. Also, featherless doesn't host the OCR vision model, so the VLM got its own `HF_VLM_PROVIDER` (auto).
My error message was also wrong: it labelled `model_not_supported` as "temporarily unavailable, retry in a minute",
which retrying can never fix.

**Switch.** `HF_MODEL=meta-llama/Llama-3.3-70B-Instruct` on `auto` (answered in 0.6 s). First calibration run:
A 86.6/86.6/84.3, B 88.1 x3, C 30.5 x3, but **Observability = 0 for both A and B**, although A lists Prometheus and Grafana
and B lists Grafana and OpenTelemetry. Cause: Llama categorised "Observability Tooling" as an *experience* criterion (Qwen had said
*skill*), and in sectioned mode the experience scorer only saw roles and dates, not the skills list. Any
miscategorised criterion loses its evidence. Since the fair A/B (entry 6) showed sectioning helps through speed and
focused rules, not through hiding sections, every section except education now also sees skills and projects.

**After the fix** (Llama-3.3-70B, sectioned, T=0, 3 repeats, profile re-extracted each run):

| Resume | Scores | Latency |
|---|---|---|
| A | 93.1, 93.1, 93.1 | ~18 s |
| B | 92.4, 92.4, 92.4 | ~18 s |
| C | 31.7, 31.7, 31.7 | ~14 s |

|A - B| = 0.7, no run-to-run change at all, no ungrounded quotes. Through the API, the text, two-column and scanned
(Tesseract) versions of A all score 93.1 with identical per-criterion levels, in 8-23 s.

## 9. Any provider, fallback models, and the right error

- `app/providers.py`: a provider registry (Hugging Face via `huggingface_hub`; OpenAI, Gemini, Groq, Mistral,
  OpenRouter, Together, Ollama and any other OpenAI-compatible endpoint via one httpx transport). Every failure is
  classified as auth / model / transient / json_mode / other.
- `LLMClient(provider, model, fallback_models)`: retries only transient errors. A model that isn't being served steps
  down to the next fallback model, and with none left it raises `MODEL_NOT_AVAILABLE` (HTTP 503, hint: change the
  config). The old "temporary, retry in a minute" message for `model_not_supported` is gone. Output from a fallback
  model is never cached under the primary model's key, and the result carries a warning.
- Live check on Hugging Face with a model name nobody serves: with a fallback it took 2 calls and 0 retries, and
  the fallback answered. Without one it raised `MODEL_NOT_AVAILABLE`. By then Qwen2.5-72B was answering on `auto` again
  (novita recovered), which fits entry 8.
- UI bug: the hero card showed the candidate summary as a raw-HTML text box. The HTML was an indented f-string, and
  when there was no knockout badge the empty line plus 4-space indentation made Markdown render the rest as a code
  block. All HTML blocks now go through a helper that strips indentation and blank lines.
