# Calibration report: Qwen/Qwen2.5-72B-Instruct

- Provider: `auto`, temperature 0, seed fixed, scorer input format: `json`, scorer temperature 0.0, scorer samples 1, scoring mode `single`
- Repeats per resume: 3; JD criteria cached per JD; profile cache off
- **|mean(A) - mean(B)| = 0.5** (requirement: similar resumes must not be 40 points apart)

| Resume | Mean | Min | Max | Range | Stdev | Ungrounded (total) | Mean latency | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A (strong) | 93.8 | 93.3 | 94.8 | 1.5 | 0.7 | 3 | 54.0 s | 4151 |
| B (strong, similar to A) | 93.3 | 87.3 | 96.3 | 9.0 | 4.2 | 0 | 56.6 s | 3875 |
| C (adjacent / weaker) | 30.1 | 27.8 | 33.5 | 5.7 | 2.5 | 0 | 47.9 s | 3593 |

Per-criterion levels, first run of A vs B:

| Criterion | A | B |
|---|---|---|
| Backend Experience | 4 | 4 |
| Python Proficiency | 4 | 4 |
| Database Experience | 4 | 4 |
| Message Queues | 4 | 4 |
| Cloud Deployment | 4 | 4 |
| Education | 3 | 3 |
| Payments Domain | 3 | 3 |
| Kubernetes Familiarity | 2 | 3 |
| Observability Tools | 3 | 3 |
