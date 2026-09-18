# Calibration report: Qwen/Qwen2.5-72B-Instruct

- Provider: `auto`, temperature 0, seed fixed, scorer input format: `markdown`, scorer temperature 0.5, scorer samples 3, scoring mode `single`
- Repeats per resume: 3; JD criteria cached per JD; profile cache off
- **|mean(A) - mean(B)| = 6.5** (requirement: similar resumes must not be 40 points apart)

| Resume | Mean | Min | Max | Range | Stdev | Ungrounded (total) | Mean latency | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A (strong) | 93.8 | 93.3 | 94.8 | 1.5 | 0.7 | 2 | 58.6 s | 8630 |
| B (strong, similar to A) | 87.3 | 86.5 | 88.8 | 2.3 | 1.1 | 2 | 70.4 s | 8203 |
| C (adjacent / weaker) | 22.0 | 19.5 | 23.2 | 3.7 | 1.7 | 1 | 40.3 s | 7556 |

Per-criterion levels, first run of A vs B:

| Criterion | A | B |
|---|---|---|
| Backend Experience | 4 | 3 |
| Python Proficiency | 4 | 4 |
| Database Experience | 4 | 4 |
| Message Queues | 4 | 3 |
| Cloud Deployment | 4 | 3 |
| Education | 3 | 3 |
| Payments Domain | 3 | 3 |
| Kubernetes Familiarity | 2 | 1 |
| Observability Tools | 3 | 3 |
