# Calibration report: Qwen/Qwen2.5-72B-Instruct

- Provider: `auto`, temperature 0, seed fixed, scorer input format: `markdown`, scorer temperature 0.0, scorer samples 1
- Repeats per resume: 3; JD criteria cached per JD; profile cache off
- **|mean(A) - mean(B)| = 2.6** (requirement: similar resumes must not be 40 points apart)

| Resume | Mean | Min | Max | Range | Stdev | Ungrounded (total) | Mean latency | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A (strong) | 90.1 | 90.1 | 90.1 | 0.0 | 0.0 | 6 | 57.6 s | 4274 |
| B (strong, similar to A) | 87.5 | 85.0 | 88.8 | 3.8 | 1.8 | 2 | 46.5 s | 3585 |
| C (adjacent / weaker) | 24.0 | 23.2 | 24.4 | 1.2 | 0.6 | 0 | 35.8 s | 3202 |

Per-criterion levels, first run of A vs B:

| Criterion | A | B |
|---|---|---|
| Backend Experience | 4 | 4 |
| Python Proficiency | 4 | 4 |
| Database Experience | 4 | 4 |
| Message Queues | 4 | 3 |
| Cloud Deployment | 4 | 3 |
| Education | 2 | 3 |
| Payments Domain | 3 | 3 |
| Kubernetes Familiarity | 1 | 1 |
| Observability Tools | 3 | 3 |
