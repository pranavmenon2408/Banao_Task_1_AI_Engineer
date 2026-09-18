# Calibration report: Qwen/Qwen2.5-72B-Instruct

- Provider: `auto`, temperature 0, seed fixed, scorer input format: `markdown`, scorer temperature 0.0, scorer samples 1, scoring mode `single`
- Repeats per resume: 5; JD criteria cached per JD; profile cache on (scorer-only variance)
- **|mean(A) - mean(B)| = 5.1** (requirement: similar resumes must not be 40 points apart)

| Resume | Mean | Min | Max | Range | Stdev | Ungrounded (total) | Mean latency | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A (strong) | 93.9 | 93.3 | 94.8 | 1.5 | 0.7 | 5 | 34.8 s | 2944 |
| B (strong, similar to A) | 88.8 | 86.5 | 91.0 | 4.5 | 1.4 | 2 | 30.2 s | 2801 |
| C (adjacent / weaker) | 28.7 | 27.8 | 32.3 | 4.5 | 1.8 | 0 | 29.6 s | 2648 |

Per-criterion levels, first run of A vs B:

| Criterion | A | B |
|---|---|---|
| Backend Experience | 4 | 4 |
| Python Proficiency | 4 | 4 |
| Database Experience | 4 | 4 |
| Message Queues | 4 | 3 |
| Cloud Deployment | 4 | 3 |
| Education | 3 | 3 |
| Payments Domain | 3 | 3 |
| Kubernetes Familiarity | 2 | 1 |
| Observability Tools | 3 | 3 |
