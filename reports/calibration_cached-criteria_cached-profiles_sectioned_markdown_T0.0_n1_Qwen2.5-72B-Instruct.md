# Calibration report: Qwen/Qwen2.5-72B-Instruct

- Provider: `auto`, temperature 0, seed fixed, scorer input format: `markdown`, scorer temperature 0.0, scorer samples 1, scoring mode `sectioned`
- Repeats per resume: 5; JD criteria cached per JD; profile cache on (scorer-only variance)
- **|mean(A) - mean(B)| = 0.6** (requirement: similar resumes must not be 40 points apart)

| Resume | Mean | Min | Max | Range | Stdev | Ungrounded (total) | Mean latency | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A (strong) | 93.3 | 93.3 | 93.3 | 0.0 | 0.0 | 5 | 23.6 s | 6354 |
| B (strong, similar to A) | 93.9 | 88.8 | 96.3 | 7.5 | 2.8 | 2 | 23.1 s | 6090 |
| C (adjacent / weaker) | 29.9 | 29.0 | 33.5 | 4.5 | 1.8 | 0 | 18.8 s | 5815 |

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
| Kubernetes Familiarity | 1 | 3 |
| Observability Tools | 3 | 3 |
