"""Calibration check: are similar resumes scored similarly, and is a single resume scored stably?

Runs every sample resume against the sample JD `--repeats` times with the profile cache OFF, so each
run re-does resume extraction and scoring (the two LLM steps that vary per resume).

  --fresh-criteria  also re-extracts JD criteria on every run (cache OFF). This is the ablation that
                    shows why criteria are cached per JD in production.

Writes a Markdown + JSON report to reports/.

    python -m scripts.calibrate --repeats 3
    python -m scripts.calibrate --repeats 3 --fresh-criteria
    python -m scripts.calibrate --repeats 2 --model meta-llama/Llama-3.3-70B-Instruct
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.documents.parsing import extract_text  # noqa: E402
from app.llm.client import LLMClient, LLMError  # noqa: E402
from app.pipeline import ScoringPipeline  # noqa: E402

SAMPLES = {
    "A (strong)": "samples/resume_a_strong.txt",
    "B (strong, similar to A)": "samples/resume_b_strong_similar.txt",
    "C (adjacent / weaker)": "samples/resume_c_adjacent.txt",
}
JD = "samples/jd_backend_engineer.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fresh-criteria", action="store_true")
    ap.add_argument(
        "--cache-profiles", action="store_true", help="reuse Agent 1's profile so only the scorer varies between runs"
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--provider", default=None, help="LLM provider (huggingface, openai, google, groq, ...)")
    ap.add_argument("--hf-provider", default=None, help="Hugging Face inference provider (auto, novita, ...)")
    ap.add_argument("--input-format", choices=["markdown", "json"], default=None)
    ap.add_argument("--scorer-temperature", type=float, default=None)
    ap.add_argument("--scorer-samples", type=int, default=None)
    ap.add_argument("--scoring-mode", choices=["single", "sectioned"], default=None)
    args = ap.parse_args()

    llm = LLMClient(model=args.model, provider=args.provider, hf_provider=args.hf_provider)
    pipe = ScoringPipeline(llm=llm, cache_criteria=not args.fresh_criteria, cache_profiles=args.cache_profiles)
    overrides = {
        "scorer_input_format": args.input_format,
        "scorer_temperature": args.scorer_temperature,
        "scorer_samples": args.scorer_samples,
        "scoring_mode": args.scoring_mode,
    }
    pipe.cfg = pipe.cfg.model_copy(update={k: v for k, v in overrides.items() if v is not None})
    fmt = pipe.cfg.scorer_input_format
    variant = f"{pipe.cfg.scoring_mode}_{fmt}_T{pipe.cfg.scorer_temperature}_n{pipe.cfg.scorer_samples}"
    jd = (ROOT / JD).read_text(encoding="utf-8")

    runs: dict[str, list[dict]] = {k: [] for k in SAMPLES}
    t_start = time.perf_counter()
    for rep in range(args.repeats):
        for label, path in SAMPLES.items():
            doc = extract_text(Path(path).name, (ROOT / path).read_bytes())
            try:
                r = pipe.run(doc.text, jd)
            except LLMError as exc:
                print(f"[{label}] run {rep + 1} failed: {exc.code.value}: {exc.message}")
                continue
            row = {
                "overall": r.overall_score,
                "recommendation": r.recommendation,
                "levels": {s.criterion.name: s.final_level for s in r.criteria},
                "raw_levels": {s.criterion.name: s.raw_level for s in r.criteria},
                "ungrounded": sum(not s.grounded for s in r.criteria),
                "mean_agreement": round(sum(s.agreement for s in r.criteria) / len(r.criteria), 2),
                "n_criteria": len(r.criteria),
                "latency_s": r.meta.total_latency_ms / 1000,
                "tokens": sum(x.prompt_tokens + x.completion_tokens for x in r.meta.stages),
                "llm_calls": sum(x.llm_calls for x in r.meta.stages),
                "retries": sum(x.retries for x in r.meta.stages),
            }
            runs[label].append(row)
            print(
                f"[{label}] run {rep + 1}: {r.overall_score:5.1f} {r.recommendation:14s} "
                f"criteria={row['n_criteria']} ungrounded={row['ungrounded']} "
                f"{row['latency_s']:.1f}s {row['tokens']} tok"
            )

    summary = {}
    for label, rs in runs.items():
        scores = [r["overall"] for r in rs]
        if not scores:
            continue
        summary[label] = {
            "mean": round(st.mean(scores), 1),
            "min": min(scores),
            "max": max(scores),
            "range": round(max(scores) - min(scores), 1),
            "stdev": round(st.pstdev(scores), 1) if len(scores) > 1 else 0.0,
            "mean_latency_s": round(st.mean(r["latency_s"] for r in rs), 1),
            "mean_tokens": round(st.mean(r["tokens"] for r in rs)),
            "ungrounded_total": sum(r["ungrounded"] for r in rs),
        }
    labels = list(SAMPLES)
    a, b, c = (summary.get(lbl, {}).get("mean") for lbl in labels)
    gap_ab = round(abs(a - b), 1) if a is not None and b is not None else None

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    tag = (
        ("fresh-criteria" if args.fresh_criteria else "cached-criteria")
        + ("_cached-profiles" if args.cache_profiles else "")
        + f"_{variant}_"
        + llm.model.split("/")[-1]
    )
    report = {
        "model": llm.model,
        "provider": llm.provider,
        "repeats": args.repeats,
        "input_format": fmt,
        "scorer_temperature": pipe.cfg.scorer_temperature,
        "scorer_samples": pipe.cfg.scorer_samples,
        "scoring_mode": pipe.cfg.scoring_mode,
        "fresh_criteria": args.fresh_criteria,
        "summary": summary,
        "gap_A_B": gap_ab,
        "wall_time_s": round(time.perf_counter() - t_start, 1),
        "runs": runs,
    }
    (out_dir / f"calibration_{tag}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    lines = [
        f"# Calibration report: {llm.model}",
        "",
        f"- Provider: `{llm.provider}`, temperature 0, seed fixed, scorer input format: `{fmt}`, "
        f"scorer temperature {pipe.cfg.scorer_temperature}, scorer samples {pipe.cfg.scorer_samples}, "
        f"scoring mode `{pipe.cfg.scoring_mode}`",
        f"- Repeats per resume: {args.repeats}; "
        f"JD criteria {'re-extracted every run' if args.fresh_criteria else 'cached per JD'}; "
        f"profile cache {'on (scorer-only variance)' if args.cache_profiles else 'off'}",
        f"- **|mean(A) - mean(B)| = {gap_ab}** (requirement: similar resumes must not be 40 points apart)",
        "",
        "| Resume | Mean | Min | Max | Range | Stdev | Ungrounded (total) | Mean latency | Mean tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, s in summary.items():
        lines.append(
            f"| {label} | {s['mean']} | {s['min']} | {s['max']} | {s['range']} | {s['stdev']} | "
            f"{s['ungrounded_total']} | {s['mean_latency_s']} s | {s['mean_tokens']} |"
        )
    if runs[labels[0]] and runs[labels[1]]:
        la, lb = runs[labels[0]][0]["levels"], runs[labels[1]][0]["levels"]
        lines += ["", "Per-criterion levels, first run of A vs B:", "", "| Criterion | A | B |", "|---|---|---|"]
        for name in la:
            lines.append(f"| {name} | {la[name]} | {lb.get(name, '-')} |")
    (out_dir / f"calibration_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
