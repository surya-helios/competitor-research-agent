from datetime import datetime, timedelta, timezone

from competitor_research_agent.memory import CompetitorMemory, memory_key
from competitor_research_agent.schemas import Candidate, CompanyProfile, ResearchConstraints

ACME = CompanyProfile(name="Acme", domain="www.acme.com")
BETA = Candidate(name="Beta", domain="beta.io", rank=1, is_direct=True, overlap_score=0.8, rationale="x")


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 8, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def test_recall_returns_what_was_approved(tmp_path):
    memory = CompetitorMemory(tmp_path / "app.sqlite")
    assert memory.recall(ACME, ResearchConstraints()) is None
    memory.remember(ACME, ResearchConstraints(), [BETA], "run-1")

    recalled = memory.recall(ACME, ResearchConstraints())
    assert recalled.competitors == [BETA] and recalled.run_id == "run-1"


def test_later_approval_replaces_earlier_one(tmp_path):
    memory = CompetitorMemory(tmp_path / "app.sqlite")
    memory.remember(ACME, ResearchConstraints(), [BETA], "run-1")
    gamma = BETA.model_copy(update={"name": "Gamma"})
    memory.remember(ACME, ResearchConstraints(), [gamma], "run-2")
    assert [c.name for c in memory.recall(ACME, ResearchConstraints()).competitors] == ["Gamma"]


def test_old_approvals_expire(tmp_path):
    clock = Clock()
    memory = CompetitorMemory(tmp_path / "app.sqlite", max_age_days=90, now=clock)
    memory.remember(ACME, ResearchConstraints(), [BETA], "run-1")
    clock.now += timedelta(days=91)
    assert memory.recall(ACME, ResearchConstraints()) is None


def test_key_uses_website_market_and_region():
    by_site = memory_key(ACME, ResearchConstraints())
    assert by_site == memory_key(CompanyProfile(name="ACME Inc", domain="acme.com"), ResearchConstraints())
    assert by_site != memory_key(ACME, ResearchConstraints(region="India"))
    assert memory_key(CompanyProfile(name="Acme"), ResearchConstraints(market="CRM")) == "acme|crm|"
