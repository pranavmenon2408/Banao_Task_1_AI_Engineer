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
