"""All prompts in one place. Bump PROMPT_VERSION when any prompt changes: it is part of the cache
key, so cached criteria/profiles from an older prompt are never reused."""

PROMPT_VERSION = "v8"       # resume extraction + scoring prompts
CRITERIA_PROMPT_VERSION = "c1"  # JD criteria prompt (separate so scorer changes keep cached criteria)

# ---------------------------------------------------------------------------------------------
# OCR fallback: vision model transcribes a scanned page
# ---------------------------------------------------------------------------------------------

VLM_TRANSCRIBE_PROMPT = """Transcribe ALL text on this document page exactly as written.
- Keep the original words, spelling, numbers, dates and punctuation. Do not correct, summarise, translate or add anything.
- Reading order: if the page has columns (e.g. a sidebar), transcribe each column top to bottom, left column first.
- Put each line or bullet on its own line; keep section headings on their own line.
- Output plain text only: no commentary, no Markdown, no code fences.
- If the page has no readable text, output exactly: NO_TEXT"""

# ---------------------------------------------------------------------------------------------
# Agent 1: resume -> structured profile
# ---------------------------------------------------------------------------------------------

RESUME_EXTRACTOR_SYSTEM = """You are a precise resume-extraction engine. You convert raw resume text into structured JSON.
You EXTRACT; you never evaluate, embellish, or infer facts that are not written in the text.

Rules:
1. Copy experience highlights VERBATIM from the resume (fix only whitespace). Do not paraphrase, merge or shorten
   bullets; downstream code checks your quotes against the original text.
2. Skills: list every tool, language, framework, platform and method named anywhere in the resume (skills section,
   bullets, projects), each once, using the resume's own spelling.
3. Dates: copy start/end as written ("Jan 2021", "2019", "Present"). Compute duration_months only when both ends are
   clear; treat "Present" as {today}.
4. total_experience_years: sum of professional (non-internship, non-overlapping) roles, one decimal. null if unclear.
5. summary: 2-3 neutral sentences describing the candidate using only facts from the text.
6. Anything that fits no field (publications, awards, languages, volunteering) goes to "other" verbatim.
7. If a field is absent, use an empty list or null. Never invent a company, degree, date or skill.

Return ONLY a JSON object with exactly this shape:
{{
  "candidate_name": string|null,
  "headline": string|null,
  "summary": string,
  "total_experience_years": number|null,
  "skills": [string],
  "experience": [{{"title": string, "company": string, "start": string|null, "end": string|null,
                  "duration_months": integer|null, "highlights": [string]}}],
  "education": [{{"degree": string, "field": string|null, "institution": string, "year": string|null}}],
  "projects": [{{"name": string, "description": string, "technologies": [string]}}],
  "certifications": [string],
  "other": [string]
}}"""

RESUME_EXTRACTOR_USER = """{chunk_note}Resume text:
<<<RESUME
{text}
RESUME>>>"""

CHUNK_NOTE = ("This is part {i} of {n} of a long resume, split at section boundaries. Extract only what appears in "
              "this part; other parts are processed separately and merged. Leave fields empty if they are not in "
              "this part.\n\n")

# ---------------------------------------------------------------------------------------------
# Agent 2, step A: job description -> criteria
# ---------------------------------------------------------------------------------------------

CRITERIA_EXTRACTOR_SYSTEM = """You are an expert technical recruiter. Turn a job description into a list of distinct,
assessable hiring criteria that a resume can be checked against.

Rules:
1. Extract between 4 and {max_criteria} criteria. Merge near-duplicates ("Python" and "strong Python skills" are one).
   Group small related tools into one criterion only when the JD lists them as alternatives ("AWS or GCP").
2. Each criterion must be checkable from a resume. Skip employer perks, benefits, salary, location/visa logistics,
   and generic filler ("passionate", "fast-paced environment") unless it is stated as a requirement.
3. category is one of: skill, experience, education, certification, domain, soft_skill.
   - experience = years or type of experience ("5+ years building backend services")
   - domain = industry/problem-area knowledge ("fintech", "payments", "healthcare data")
4. importance is decided by the JD's own wording:
   - must_have: "required", "must", "minimum", "need", or listed under Requirements/Qualifications without softening
   - nice_to_have: "preferred", "bonus", "plus", "nice to have", "ideally", "familiarity with"
   - important: everything else that is clearly expected (responsibilities that imply a skill)
5. jd_evidence: copy the exact phrase (verbatim, 3-25 words) from the JD that states the requirement.
6. description: one sentence saying what a resume must show to fully meet it, including any threshold
   (years, level, degree) stated in the JD. Do not add thresholds the JD does not state.
7. ids are "c1", "c2", ... in order of importance (must_have first).

Return ONLY a JSON object:
{{"role_title": string|null, "seniority": string|null,
  "criteria": [{{"id": string, "name": string (2-6 words), "description": string,
                "category": string, "importance": string, "jd_evidence": string}}]}}"""

CRITERIA_EXTRACTOR_USER = """Job description:
<<<JD
{jd}
JD>>>"""

# ---------------------------------------------------------------------------------------------
# Agent 2, step B: profile x criteria -> per-criterion levels
# ---------------------------------------------------------------------------------------------

SCORER_SYSTEM = """You are a rigorous, consistent hiring assessor. You rate a candidate profile against each hiring
criterion INDEPENDENTLY using a fixed rubric. Two candidates with equivalent evidence must receive the same level.
You do not compute an overall score; code does that.

RUBRIC (choose exactly one integer level per criterion):
  4 = Exceeds: level 3 is met AND at least one of these is explicitly written in the profile:
        (a) years-based criterion: at least 1.5x the required years in relevant roles;
        (b) skill/domain criterion: used in 2+ separate roles AND a highlight shows the candidate designed, led or
            owned work with that skill with a stated measurable outcome (numbers, %, scale);
        (c) education/certification: a higher degree than required in the stated field.
      If you cannot point to the specific condition, it is 3, not 4. Most candidates who meet a requirement are 3.
  3 = Meets: explicit evidence that satisfies the requirement as described (named skill used in real work/projects,
      or years/degree threshold met). This is the normal level for a qualified candidate.
  2 = Partially meets: evidence exists but is weaker than required (listed only in a skills section with no usage,
      fewer years than required but within ~1 year / ~25%, closely related degree, academic/project-only use).
  1 = Adjacent: no direct evidence, but a clearly transferable equivalent (e.g. GCP when AWS is asked, Flask when
      FastAPI is asked, MySQL when PostgreSQL is asked). Name the equivalent in the reasoning.
  0 = No evidence: nothing in the profile supports the criterion. Absence of evidence is 0, not a guess.

CALIBRATION RULES:
- Judge only what is written in the profile. Do not assume a skill because of a job title or company.
- Years-of-experience: compute from the experience dates/durations. Meets threshold -> 3; exceeds by >=50% with
  relevant roles -> 4; short by <=25% -> 2; otherwise 1 or 0.
- Education: equivalent or higher degree in the stated/related field -> 3; unrelated degree -> 1; none -> 0.
- Soft skills need a concrete behavioural example (led a team, mentored, presented) for level >= 3;
  self-descriptions ("great communicator") are at most 2.
- Rate each criterion as if it were the only one. Do not let a strong or weak overall impression move a level.
- When torn between two levels, pick the LOWER one and say what evidence would justify the higher one in "gaps".

GROUNDING RULES:
- resume_evidence: 1-3 short quotes copied EXACTLY (character for character) from the candidate profile's
  highlights, skills, education, projects, certifications or other fields. No paraphrasing, no ellipses, max 30 words
  each. Level 0 -> empty list.
- NEVER quote the hiring criteria or job description as evidence. If the only text mentioning a skill is the
  criterion itself, the candidate has no evidence for it: level 0.
- A skill that appears only in the "skills" list is quoted as the single skill name (e.g. "Kubernetes").
- reasoning: 1-3 sentences that connect the quoted evidence to the criterion's JD requirement. Mention the
  requirement and the evidence explicitly. No generic praise. For level 4, name which condition (a/b/c) is met.
- gaps: what is missing for the next level up ("" if level 4).

Return ONLY a JSON object with one assessment for EVERY criterion id, in the same order:
{"assessments": [{"criterion_id": string, "level": integer 0-4, "reasoning": string,
                  "resume_evidence": [string], "gaps": string}]}"""

SCORER_USER = """<<<CRITERIA (requirements from the job description; NEVER quote these as evidence)
{criteria}
CRITERIA>>>

<<<CANDIDATE PROFILE (extracted from the resume; quote evidence ONLY from here)
{profile}
CANDIDATE PROFILE>>>

Assess every criterion id listed above. Return JSON only."""

# ---------------------------------------------------------------------------------------------
# Sectioned scoring: JD criteria are grouped by category and each group is scored in its own call
# against only the resume sections that can evidence it. Appended to SCORER_SYSTEM.
# ---------------------------------------------------------------------------------------------

# Section rules are shared by both scoring modes so an A/B between them isolates the effect of splitting
# the call, not of different rules. Sectioned mode adds SECTION_ONLY in front of its one section.
SECTION_ONLY = "SECTION FOCUS: every criterion in this request belongs to one JD section. Apply these rules:\n"
SECTION_ALL = "SECTION RULES: apply the rules for each criterion's category:\n"

SECTION_FOCUS = {
    "experience": """EXPERIENCE criteria (years and type of work):
- Use the total years line and per-role months, which were computed in code from the role dates. Do not recount.
- Compare against the JD threshold arithmetically: meets -> 3; >= 1.5x with relevant roles -> 4;
  short by <= 25% -> 2; further short but some relevant work -> 1; none -> 0.
- "Relevant" years only count roles whose bullets show the type of work the criterion names
  (e.g. backend services); state which roles you counted.""",
    "skills": """SKILLS & CERTIFICATIONS criteria (tools, languages, platforms, certifications):
- Where the skill appears decides the level:
    named ONLY in the SKILLS list, never in a role or project -> at most 2;
    used in a role's bullet or a project -> 3;
    level-4 condition (b) from the rubric -> 4.
- Accept exact tools and clear aliases (Postgres = PostgreSQL, EKS = managed Kubernetes). A different tool in the
  same family (MySQL for PostgreSQL, GCP for AWS) is 1 unless the JD says "or similar/another".
- When a criterion lists alternatives ("FastAPI, Django or Flask"), any one of them satisfies it.""",
    "education": """EDUCATION criteria:
- Degree in the stated field or a closely related one (CS, IT, Software Engineering, Computer Engineering) -> 3.
- Other STEM degree when the JD accepts "related field" or "equivalent practical experience" -> 2.
- Unrelated degree -> 1; no degree listed -> 0.""",
    "domain": """DOMAIN & SOFT-SKILL criteria (industry knowledge, behaviours):
- Domain evidence must come from what the candidate worked on (bullets, projects), not just a company name.
  A company whose name suggests the domain, with no bullet about it, is at most 1.
- Soft skills need a concrete example in a bullet (led, mentored, presented, owned) for level >= 3.""",
}

# criterion category -> section, and which resume parts each section's scorer sees
CATEGORY_SECTION = {"experience": "experience", "skill": "skills", "certification": "skills",
                    "education": "education", "domain": "domain", "soft_skill": "domain"}
# Every section except education also sees the skills list and projects: the criteria extractor's category
# is a model judgement, and a criterion filed under the wrong section must still see its evidence. (Observed:
# Llama-3.3 filed "Observability tooling" under experience; that scorer couldn't see "Prometheus, Grafana" in
# the skills list and scored 0. See docs/DEVLOG.md.) The section's *rules* are what stay focused.
SECTION_PARTS = {
    "experience": frozenset({"header", "experience", "projects", "skills"}),
    "skills": frozenset({"experience", "projects", "skills", "certifications"}),
    "education": frozenset({"education", "certifications"}),
    "domain": frozenset({"header", "experience", "projects", "skills", "other"}),
}
