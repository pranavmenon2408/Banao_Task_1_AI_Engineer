# Calibration report: Qwen/Qwen2.5-72B-Instruct

- Provider: `auto`, temperature 0, seed fixed, scorer input format: `markdown`, scorer temperature 0.0, scorer samples 1, scoring mode `sectioned`
- Repeats per resume: 3; JD criteria cached per JD; profile cache off
- **|mean(A) - mean(B)| = 2.5** (requirement: similar resumes must not be 40 points apart)

| Resume | Mean | Min | Max | Range | Stdev | Ungrounded (total) | Mean latency | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A (strong) | 93.8 | 93.3 | 94.8 | 1.5 | 0.7 | 3 | 45.0 s | 7677 |
| B (strong, similar to A) | 91.3 | 90.3 | 91.8 | 1.5 | 0.7 | 0 | 42.1 s | 7283 |
| C (adjacent / weaker) | 30.0 | 29.0 | 32.1 | 3.1 | 1.5 | 0 | 33.6 s | 6792 |

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
| Kubernetes Familiarity | 1 | 3 |
| Observability Tools | 3 | 3 |
