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
