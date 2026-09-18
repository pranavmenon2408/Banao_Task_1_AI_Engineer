from app.agents.scoring_agent import _jobs, _vote
from app.schemas import (Criterion, CriterionAssessment, EducationItem, ExperienceItem, ResumeProfile)

PROFILE = ResumeProfile(
    candidate_name="A", skills=["Kafka", "Python"],
    experience=[ExperienceItem(title="Engineer", company="Pay", start="Jan 2020", end="Dec 2022",
                               highlights=["Built Kafka consumers in Python"])],
    education=[EducationItem(degree="B.Tech", field="Computer Science", institution="IIT")])


def crit(cid, category):
    return Criterion(id=cid, name=cid, description="d", category=category, importance="must_have")


def test_single_mode_is_one_call_with_everything():
    jobs = _jobs(PROFILE, [crit("c1", "skill"), crit("c2", "education")], "single", "markdown")
    assert len(jobs) == 1 and "## EDUCATION" in jobs[0][1] and "## SKILLS" in jobs[0][1]


def test_sectioned_routes_criteria_to_matching_resume_sections():
    cs = [crit("c1", "experience"), crit("c2", "skill"), crit("c3", "certification"), crit("c4", "education"),
          crit("c5", "domain")]
    jobs = {tuple(c.id for c in crits): (system, text) for system, text, crits in _jobs(PROFILE, cs, "sectioned", "markdown")}
    assert set(jobs) == {("c1",), ("c2", "c3"), ("c4",), ("c5",)}
    exp_sys, exp_text = jobs[("c1",)]
    assert "SECTION FOCUS: EXPERIENCE" in exp_sys and "computed in code" in exp_text and "## SKILLS" not in exp_text
    _, skills_text = jobs[("c2", "c3")]
    assert "## SKILLS" in skills_text and "## EXPERIENCE" in skills_text and "## EDUCATION" not in skills_text
    _, edu_text = jobs[("c4",)]
    assert "## EDUCATION" in edu_text and "## EXPERIENCE" not in edu_text


def test_median_vote_and_agreement():
    a = lambda lvl: {"c1": CriterionAssessment(criterion_id="c1", level=lvl, reasoning=f"r{lvl}")}
    out = _vote([a(3), a(4), a(3)], ["c1"])
    assert out["c1"].level == 3 and out["c1"].agreement == 0.67 and out["c1"].reasoning == "r3"
