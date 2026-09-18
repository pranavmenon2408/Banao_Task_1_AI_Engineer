# Calibration report: meta-llama/Llama-3.3-70B-Instruct

- Provider: `auto`, temperature 0, seed fixed, scorer input format: `markdown`, scorer temperature 0.0, scorer samples 1, scoring mode `sectioned`
- Repeats per resume: 3; JD criteria cached per JD; profile cache off
- **|mean(A) - mean(B)| = 0.7** (requirement: similar resumes must not be 40 points apart)

| Resume | Mean | Min | Max | Range | Stdev | Ungrounded (total) | Mean latency | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A (strong) | 93.1 | 93.1 | 93.1 | 0.0 | 0.0 | 0 | 18.6 s | 7581 |
| B (strong, similar to A) | 92.4 | 92.4 | 92.4 | 0.0 | 0.0 | 0 | 18.1 s | 7322 |
| C (adjacent / weaker) | 31.7 | 31.7 | 31.7 | 0.0 | 0.0 | 0 | 14.1 s | 6769 |

Per-criterion levels, first run of A vs B:

| Criterion | A | B |
|---|---|---|
| Backend Experience | 4 | 4 |
| Python Skills | 4 | 4 |
| PostgreSQL Experience | 4 | 4 |
| Cloud Deployment | 3 | 3 |
| Computer Science Degree | 3 | 3 |
| Message Queue Experience | 4 | 3 |
| Payments Experience | 4 | 4 |
| Kubernetes Familiarity | 2 | 3 |
| Observability Tooling | 3 | 3 |
