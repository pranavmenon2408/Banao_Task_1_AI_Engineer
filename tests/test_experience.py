from datetime import date

from app.scoring.experience import parse_month, total_years
from app.core.schemas import ExperienceItem

TODAY = date(2026, 9, 1)


def role(start, end, title="Engineer"):
    return ExperienceItem(title=title, company="X", start=start, end=end)


def test_parse_formats():
    assert parse_month("Mar 2022") == 2022 * 12 + 2
    assert parse_month("06/2020") == 2020 * 12 + 5
    assert parse_month("2019") == 2019 * 12
    assert parse_month("2019", end=True) == 2019 * 12 + 11
    assert parse_month("Present", today=TODAY) == 2026 * 12 + 8
    assert parse_month("sometime") is None


def test_total_years_sequential_roles():
    exp = [role("Jul 2019", "Feb 2022"), role("Mar 2022", "Present")]
    assert total_years(exp, today=TODAY) == 7.2


def test_overlap_not_double_counted_and_interns_excluded():
    exp = [role("Jan 2020", "Dec 2021"), role("Jan 2021", "Dec 2022"),
           role("Jun 2019", "Aug 2019", title="Software Engineering Intern")]
    assert total_years(exp, today=TODAY) == 3.0


def test_unparseable_returns_none():
    assert total_years([role(None, None)]) is None
