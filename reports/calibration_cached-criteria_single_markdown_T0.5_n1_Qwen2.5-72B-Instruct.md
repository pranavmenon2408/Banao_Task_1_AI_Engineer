# Calibration report: Qwen/Qwen2.5-72B-Instruct

- Provider: `auto`, temperature 0, seed fixed, scorer input format: `markdown`, scorer temperature 0.5, scorer samples 1, scoring mode `single`
- Repeats per resume: 3; JD criteria cached per JD; profile cache off
- **|mean(A) - mean(B)| = 3.0** (requirement: similar resumes must not be 40 points apart)

| Resume | Mean | Min | Max | Range | Stdev | Ungrounded (total) | Mean latency | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A (strong) | 89.3 | 80.7 | 93.9 | 13.2 | 6.1 | 2 | 83.6 s | 3811 |
| B (strong, similar to A) | 86.3 | 85.0 | 87.3 | 2.3 | 1.0 | 3 | 89.4 s | 3564 |
| C (adjacent / weaker) | 25.9 | 24.4 | 29.0 | 4.6 | 2.2 | 0 | 54.4 s | 3181 |

Per-criterion levels, first run of A vs B:

| Criterion | A | B |
|---|---|---|
| Backend Experience | 4 | 3 |
| Python Proficiency | 4 | 4 |
| Database Experience | 4 | 4 |
| Message Queues | 4 | 3 |
| Cloud Deployment | 4 | 4 |
| Education | 3 | 3 |
| Payments Domain | 3 | 3 |
| Kubernetes Familiarity | 1 | 1 |
| Observability Tools | 3 | 2 |
