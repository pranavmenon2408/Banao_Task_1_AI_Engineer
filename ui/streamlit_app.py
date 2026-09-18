"""Streamlit front end for the Resume-JD Fit Scorer.

The UI talks to the FastAPI backend over HTTP only, so the two can be deployed and scaled independently
(see docker-compose.yml). Weight changes in the sidebar call `/api/v1/rescore`, which re-aggregates the existing
per-criterion results without another LLM call.
"""

from __future__ import annotations

import html
import json
import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 300

st.set_page_config(page_title="Fit Scorer", page_icon="🎯", layout="wide")

IMPORTANCE = {
    "must_have": ("Must have", "#b42318", "#fef3f2"),
    "important": ("Important", "#b54708", "#fffaeb"),
    "nice_to_have": ("Nice to have", "#175cd3", "#eff8ff"),
}
LEVEL_TEXT = {0: "No evidence", 1: "Adjacent", 2: "Partial", 3: "Meets", 4: "Exceeds"}
LEVEL_COLOR = {0: "#d92d20", 1: "#f79009", 2: "#eaaa08", 3: "#12b76a", 4: "#067647"}
REC_COLOR = {"Strong fit": "#067647", "Potential fit": "#12b76a", "Weak fit": "#f79009", "Not a fit": "#d92d20"}

st.markdown(
    """
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .card {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
         background: rgba(255,255,255,.02);}
  .hero {display:flex; gap:28px; align-items:center; flex-wrap:wrap;}
  .ring {--p:0; --c:#12b76a; width:150px; height:150px; border-radius:50%; flex:none;
         background: conic-gradient(var(--c) calc(var(--p)*1%), rgba(128,128,128,.18) 0);
         display:flex; align-items:center; justify-content:center;}
  .ring > div {width:118px; height:118px; border-radius:50%; background: #fcfcfd;
               display:flex; flex-direction:column; align-items:center; justify-content:center;}
  .ring .num {font-size:2.4rem; font-weight:700; line-height:1;}
  .ring .of {font-size:.8rem; opacity:.6;}
  .pill {display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:600; margin-right:6px;}
  .muted {opacity:.65; font-size:.88rem;}
  .crit-head {display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;}
  .crit-title {font-weight:650; font-size:1.05rem;}
  .bar {display:flex; gap:4px; margin:10px 0 4px;}
  .bar span {flex:1; height:8px; border-radius:4px; background: rgba(128,128,128,.2);}
  .quote {border-left:3px solid; padding:6px 10px; margin:6px 0; border-radius:0 8px 8px 0; font-size:.9rem;
          background: rgba(128,128,128,.07);}
  .jdq {border-color:#7a5af8;}
  .ok {border-color:#12b76a;} .bad {border-color:#d92d20;}
  .flag {color:#b54708; font-size:.85rem;}
  .kpi {font-size:1.6rem; font-weight:700;} .kpi-l {font-size:.8rem; opacity:.65; text-transform:uppercase; letter-spacing:.04em;}
</style>""",
    unsafe_allow_html=True,
)


def esc(s: object) -> str:
    """HTML-escape any value for safe interpolation into markup."""
    return html.escape(str(s or ""))


def show_html(block: str) -> None:
    """Render an HTML snippet with st.markdown.

    Markdown treats a blank line followed by lines indented 4+ spaces as a code block, so an empty optional piece
    of a template would make the rest of it render as raw text. Stripping indentation and blank lines keeps the
    snippet a single HTML block.
    """
    st.markdown("\n".join(line.strip() for line in block.splitlines() if line.strip()), unsafe_allow_html=True)


# ------------------------------------------------------------------ API helpers


@st.cache_data(ttl=30, show_spinner=False)
def api_health() -> dict[str, Any] | None:
    """The API's /health payload, or None if it is unreachable."""
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def api_config() -> dict[str, Any] | None:
    """The scoring configuration from the API, or None if it is unreachable."""
    try:
        return requests.get(f"{API_URL}/api/v1/config", timeout=5).json()
    except requests.RequestException:
        return None


def show_error(resp: requests.Response | None, exc: Exception | None = None) -> None:
    """Show an API error (code, message and hint) or a connection failure."""
    if exc is not None:
        st.error(f"**Could not reach the API at {API_URL}.** Is the backend running?\n\n`{exc}`")
        return
    try:
        err = resp.json().get("error", {})
    except ValueError:
        err = {}
    where = {"resume": "Resume", "jd": "Job description"}.get(err.get("field"), err.get("field") or "")
    title = f"{where}: " if where else ""
    msg = f"**{title}{esc(err.get('message', resp.text[:300]))}**"
    if err.get("hint"):
        msg += f"\n\n💡 {err['hint']}"
    msg += f"\n\n`{err.get('code', resp.status_code)}`"
    (st.warning if resp.status_code == 422 else st.error)(msg)


# ------------------------------------------------------------------ sidebar

health = api_health()
cfg = api_config()
with st.sidebar:
    st.markdown("### ⚙️ Backend")
    if health:
        st.success(f"API online · `{health['model'].split('/')[-1]}` via `{health['provider']}`", icon="✅")
        if not health.get("llm_configured"):
            st.warning(f"LLM not configured on the server: {health.get('config_problem') or 'check .env'}")
    else:
        st.error(f"API unreachable at {API_URL}")

    st.markdown("### ⚖️ Scoring weights")
    st.caption("Defaults come from `config/scoring.yaml`. Changing them re-scores instantly; no LLM call is made.")
    overrides = {"importance_weights": {}, "category_weights": {}}
    if cfg:
        with st.expander("Importance weights", expanded=True):
            for k, v in cfg["importance_weights"].items():
                overrides["importance_weights"][k] = st.slider(
                    IMPORTANCE.get(k, (k,))[0], 0.0, 5.0, float(v), 0.5, key=f"iw_{k}"
                )
        with st.expander("Category weights"):
            for k, v in cfg["category_weights"].items():
                overrides["category_weights"][k] = st.slider(
                    k.replace("_", " ").title(), 0.0, 3.0, float(v), 0.1, key=f"cw_{k}"
                )
        changed = (
            overrides["importance_weights"] != cfg["importance_weights"]
            or overrides["category_weights"] != cfg["category_weights"]
        )
    else:
        changed = False

    st.markdown("### ℹ️ How scores work")
    st.caption(
        "Each criterion gets a rubric level 0-4 from the scorer, which must quote the resume. "
        "Quotes are verified against the resume text; unverified evidence loses a level. "
        "The overall score is a weighted average computed in code."
    )

# ------------------------------------------------------------------ inputs

st.markdown("## 🎯 Resume ↔ Job Fit Scorer")
st.caption("Upload a job description and a resume to get a grounded, per-criterion fit assessment.")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("#### 📄 Job description")
    mode = st.radio("JD input", ["Upload file", "Paste text"], horizontal=True, label_visibility="collapsed")
    jd_file, jd_text = None, None
    if mode == "Upload file":
        jd_file = st.file_uploader(
            "JD file", type=["pdf", "docx", "txt"], key="jd_file", label_visibility="collapsed", help="PDF, DOCX or TXT"
        )
    else:
        jd_text = st.text_area(
            "JD text", height=210, placeholder="Paste the full job description here…", label_visibility="collapsed"
        )
        if jd_text:
            st.caption(f"{len(jd_text):,} characters")
with c2:
    st.markdown("#### 👤 Resume")
    resume = st.file_uploader(
        "Resume file",
        type=["pdf", "docx", "txt"],
        key="resume",
        label_visibility="collapsed",
        help="PDF, DOCX or TXT. Scanned/image-only PDFs are detected and rejected with advice.",
    )
    st.caption("Resumes are upload-only, so the text is exactly what the candidate submitted.")

ready = resume is not None and (jd_file is not None or (jd_text and len(jd_text.strip()) >= 100))
go = st.button("Assess fit", type="primary", disabled=not ready or not health, width="stretch")
if not ready:
    st.caption("Add a job description (file, or at least 100 characters of text) and a resume to continue.")

if go:
    files = {"resume": (resume.name, resume.getvalue(), resume.type or "application/octet-stream")}
    data = {}
    if jd_file is not None:
        files["jd_file"] = (jd_file.name, jd_file.getvalue(), jd_file.type or "application/octet-stream")
    else:
        data["jd_text"] = jd_text
    if changed:
        data["weights"] = json.dumps(overrides)
    with st.status("Assessing fit…", expanded=True) as status:
        st.write("Parsing documents and extracting JD criteria…")
        st.write("Agent 1 is structuring the resume; Agent 2 is scoring each criterion…")
        try:
            r = requests.post(f"{API_URL}/api/v1/score", files=files, data=data, timeout=TIMEOUT)
        except requests.RequestException as exc:
            status.update(label="Failed", state="error")
            show_error(None, exc)
            st.stop()
        if r.status_code != 200:
            status.update(label="Could not assess", state="error")
            show_error(r)
            st.stop()
        st.session_state["result"] = r.json()
        st.session_state["base_criteria"] = r.json()["criteria"]
        status.update(
            label=f"Done in {r.json()['meta']['total_latency_ms'] / 1000:.1f}s", state="complete", expanded=False
        )

res = st.session_state.get("result")
if not res:
    st.stop()

# Re-weight without re-running the LLM
if changed and st.session_state.get("base_criteria"):
    rr = requests.post(
        f"{API_URL}/api/v1/rescore",
        json={"criteria": st.session_state["base_criteria"], "overrides": overrides},
        timeout=15,
    )
    if rr.ok:
        res = {**res, **rr.json()}

# ------------------------------------------------------------------ results: hero

st.divider()
score = res["overall_score"]
rec = res["recommendation"]
color = REC_COLOR.get(rec, "#12b76a")
crit = res["criteria"]
must = [c for c in crit if c["criterion"]["importance"] == "must_have"]
must_met = sum(c["final_level"] >= 3 for c in must)
ungrounded = sum(not c["grounded"] for c in crit)

show_html(f"""
<div class="card hero">
  <div class="ring" style="--p:{score}; --c:{color};"><div><div class="num">{score:.0f}</div><div class="of">/ 100</div></div></div>
  <div style="flex:1; min-width:260px;">
    <span class="pill" style="background:{color}22; color:{color}; font-size:.95rem;">{esc(rec)}</span>
    {'<span class="pill" style="background:#fef3f2;color:#b42318;">Must-have knockout</span>' if res["knockout_triggered"] else ""}
    <h3 style="margin:.5rem 0 .2rem;">{esc(res.get("candidate_name") or "Candidate")}</h3>
    <div class="muted">for <b>{esc(res.get("role_title") or "the role")}</b></div>
    <div class="muted" style="margin-top:.6rem;">{esc(res["profile"].get("summary"))}</div>
  </div>
</div>""")

k = st.columns(4)
for col, (val, label) in zip(
    k,
    [
        (f"{must_met}/{len(must)}", "Must-haves met"),
        (f"{sum(c['final_level'] >= 3 for c in crit)}/{len(crit)}", "Criteria met"),
        (str(ungrounded), "Unverified evidence"),
        (f"{res['meta']['total_latency_ms'] / 1000:.1f}s", "Latency"),
    ],
    strict=True,
):
    col.markdown(
        f"<div class='card' style='text-align:center'><div class='kpi'>{val}</div><div class='kpi-l'>{label}</div></div>",
        unsafe_allow_html=True,
    )

if changed:
    st.info("Showing scores re-weighted with your sidebar weights (per-criterion judgements unchanged).", icon="⚖️")
if res["knockout_triggered"]:
    st.warning(
        f"A must-have criterion has no evidence, so the overall score is capped at {res['weights_used']['knockout']['cap']}."
    )
for w in res["meta"].get("warnings", []):
    st.caption(f"⚠️ {w}")

s1, s2 = st.columns(2, gap="large")
with s1:
    st.markdown("#### ✅ Strengths")
    for s in res["strengths"] or ["No criterion clearly met."]:
        st.markdown(f"- {esc(s)}")
with s2:
    st.markdown("#### ⚠️ Gaps")
    for g in res["gaps"] or ["No significant gaps on required criteria."]:
        st.markdown(f"- {esc(g)}")

tab1, tab2, tab3 = st.tabs(["📊 Per-criterion assessment", "👤 Extracted profile", "⏱️ Run metrics"])

# ------------------------------------------------------------------ per-criterion

with tab1:
    order = {"must_have": 0, "important": 1, "nice_to_have": 2}
    f1, f2 = st.columns([2, 1])
    only_issues = f2.toggle("Only gaps & flags", value=False)
    f1.caption(
        "Each score cites the JD requirement it came from and the resume text that supports it. "
        "✓ = quote verified in the resume, ✗ = not found (level reduced)."
    )
    for c in sorted(res["criteria"], key=lambda c: (order[c["criterion"]["importance"]], c["criterion"]["id"])):
        if only_issues and c["final_level"] >= 3 and not c["flags"]:
            continue
        cr = c["criterion"]
        imp_label, imp_fg, imp_bg = IMPORTANCE[cr["importance"]]
        lvl = c["final_level"]
        bars = "".join(
            f"<span style='background:{LEVEL_COLOR[lvl]}'></span>" if i < lvl else "<span></span>" for i in range(4)
        )
        adj = f" <span class='muted'>(scorer said {c['raw_level']}, reduced)</span>" if c["raw_level"] != lvl else ""
        agree = f" · agreement {c['agreement']:.0%}" if c.get("agreement", 1) < 1 else ""
        jd_quote = (
            f"<div class='quote jdq'><span class='muted'>JD says:</span> “{esc(cr['jd_evidence'])}”</div>"
            if cr.get("jd_evidence")
            else ""
        )
        ev = "".join(
            f"<div class='quote {'ok' if e['found'] else 'bad'}'>{'✓' if e['found'] else '✗'} “{esc(e['quote'])}”"
            f" <span class='muted'>({e['similarity']:.0%} match)</span></div>"
            for e in c["evidence"]
        )
        if not c["evidence"]:
            ev = "<div class='muted'>No supporting evidence found in the resume.</div>"
        flags = "".join(f"<div class='flag'>⚑ {esc(f)}</div>" for f in c["flags"])
        gaps = (
            f"<div style='margin-top:8px'><b>To reach the next level:</b> {esc(c['gaps'])}</div>"
            if c["gaps"] and lvl < 4
            else ""
        )
        show_html(f"""
<div class="card">
  <div class="crit-head">
    <div>
      <div class="crit-title">{esc(cr["name"])}</div>
      <span class="pill" style="background:{imp_bg}; color:{imp_fg};">{imp_label}</span>
      <span class="pill" style="background:rgba(128,128,128,.12);">{esc(cr["category"].replace("_", " "))}</span>
    </div>
    <div style="text-align:right; min-width:170px;">
      <div style="font-weight:700; color:{LEVEL_COLOR[lvl]};">{LEVEL_TEXT[lvl]} · {lvl}/4{adj}</div>
      <div class="muted">{c["points"]:.0f} pts × weight {c["weight"]:.2f} → +{c["weighted_contribution"]:.1f}{agree}</div>
    </div>
  </div>
  <div class="bar">{bars}</div>
  <div class="muted">{esc(cr["description"])}</div>
  {jd_quote}
  <div style="margin-top:8px">{esc(c["reasoning"])}</div>
  {ev}
  {gaps}
  {flags}
</div>""")

# ------------------------------------------------------------------ profile

with tab2:
    p = res["profile"]
    a, b = st.columns([1, 1], gap="large")
    with a:
        st.markdown(f"**{esc(p.get('candidate_name') or '')}** · {esc(p.get('headline') or '')}")
        if p.get("total_experience_years") is not None:
            st.markdown(f"**Experience:** {p['total_experience_years']} years")
        st.markdown("**Skills**")
        st.markdown(
            " ".join(
                f"<span class='pill' style='background:rgba(128,128,128,.12);margin-bottom:6px'>{esc(s)}</span>"
                for s in p.get("skills", [])
            ),
            unsafe_allow_html=True,
        )
        if p.get("education"):
            st.markdown("**Education**")
            for e in p["education"]:
                st.markdown(
                    "- "
                    + ", ".join(
                        esc(x) for x in (e.get("degree"), e.get("field"), e.get("institution"), e.get("year")) if x
                    )
                )
        if p.get("certifications"):
            st.markdown("**Certifications**")
            for c in p["certifications"]:
                st.markdown(f"- {esc(c)}")
    with b:
        st.markdown("**Experience**")
        for e in p.get("experience", []):
            dates = " – ".join(x for x in (e.get("start"), e.get("end")) if x)
            with st.expander(f"{e.get('title')} · {e.get('company')}  ({dates})"):
                for h in e.get("highlights", []):
                    st.markdown(f"- {esc(h)}")
        if p.get("projects"):
            st.markdown("**Projects**")
            for pr in p["projects"]:
                st.markdown(f"- **{esc(pr.get('name'))}**: {esc(pr.get('description'))}")

# ------------------------------------------------------------------ metrics

with tab3:
    m = res["meta"]
    st.caption(
        f"Run `{m['run_id']}` · model `{m['model']}` via `{m['provider']}` · "
        f"resume {m['resume_chars']:,} chars in {m['resume_chunks']} chunk(s)"
    )
    if m.get("extraction"):
        st.caption(
            "Text extraction: " + " · ".join(f"{k.replace('_', ' ')} → `{v}`" for k, v in m["extraction"].items())
        )
    rows = [
        {
            "Stage": s["name"].replace("_", " "),
            "Latency (s)": round(s["latency_ms"] / 1000, 2),
            "LLM calls": s["llm_calls"],
            "Prompt tok": s["prompt_tokens"],
            "Completion tok": s["completion_tokens"],
            "Retries": s["retries"],
            "Cache hit": "✓" if s["cache_hit"] else "",
            "Detail": s.get("detail") or "",
        }
        for s in m["stages"]
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    tot_tok = sum(s["prompt_tokens"] + s["completion_tokens"] for s in m["stages"])
    st.caption(f"Total tokens: {tot_tok:,} · Total LLM calls: {sum(s['llm_calls'] for s in m['stages'])}")
    with st.expander("Weights used"):
        st.json(res["weights_used"])

st.download_button(
    "⬇️ Download full assessment (JSON)",
    json.dumps(res, indent=2),
    file_name=f"fit_assessment_{res['meta']['run_id']}.json",
    mime="application/json",
)
