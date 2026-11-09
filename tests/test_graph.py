import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeLedger, FakeLLM, FakeTavily
from langgraph.graph import END

from competitor_research_agent.agents.brief_writer import BriefDraft
from competitor_research_agent.config import Settings
from competitor_research_agent.graph import fan_out, open_checkpointer
from competitor_research_agent.inputs import InvalidInput
from competitor_research_agent.memory import CompetitorMemory
from competitor_research_agent.runner import ResearchRunner
from competitor_research_agent.schemas import (
    Candidate,
    CandidateDraft,
    CandidateRanking,
    ClaimDraft,
    ClaimDrafts,
    CompanyProfile,
    ComparisonMatrix,
    DiscoveryPlan,
    EvidenceVerdict,
    PlannedQuery,
    ResearchConstraints,
    ResearchPlan,
)

US = CompanyProfile(name="Us", domain="us.com", category="docs", description="A docs tool.")


def _candidate(name, direct=True, score=0.8):
    return CandidateDraft(
        name=name, domain=f"{name.lower()}.com", is_direct=direct, overlap_score=score,
        rationale="Same buyers.", source_labels=["S1", "S9"],
    )


def _competitor_in(messages) -> str:
    return re.search(r"Competitor: (.+)", messages[-1][1]).group(1).strip()


def _research_plan(messages):
    name = _competitor_in(messages)
    return ResearchPlan(queries=[PlannedQuery(query=f"{name} pricing", purpose="pricing", use_competitor_domain=True)])


def _claims(messages):
    """One verifiable claim per required field, quoting the first evidence item."""
    evidence_id = re.search(r"\[(ev-[^\]]+)\]", messages[-1][1]).group(1)
    return ClaimDrafts(claims=[
        ClaimDraft(field=field, item=item, value=value, status="found", quote="Pro $10", evidence_id=evidence_id)
        for field, item, value in [
            ("pricing", "Pro", "$10 per user/month"),
            ("feature", "Pro plan", "Has a Pro plan"),
            ("positioning", "positioning", "Sells a Pro plan"),
        ]
    ])


def _search_results(query):
    """Every result names whatever the query names, as real pages would."""
    slug = re.sub(r"\W+", "-", query.lower())
    return [{"url": f"https://src.example/{slug}", "content": f"{query}: Pro $10"}]


def _outputs(overrides=None):
    outputs = {
        CompanyProfile: US,
        DiscoveryPlan: DiscoveryPlan(queries=["us competitors"]),
        CandidateRanking: CandidateRanking(candidates=[
            _candidate("Us"),  # the company itself: must be filtered out
            _candidate("Delta", direct=False, score=0.9),
            _candidate("Alpha", score=0.9),
            _candidate("Bravo", score=0.7),
            _candidate("Charlie", score=0.6),
        ]),
        ResearchPlan: _research_plan,
        ClaimDrafts: _claims,
        EvidenceVerdict: EvidenceVerdict(gaps=[], conflicting_labels=[], wrong_company_labels=[]),
        ComparisonMatrix: ComparisonMatrix(rows=[], implications=["Watch pricing."]),
        BriefDraft: BriefDraft(title="Brief", executive_summary="Prices are close [C1].", body_markdown="Body", gaps=[]),
    }
    outputs.update(overrides or {})
    return outputs


@pytest.fixture
def make_runner(tmp_path):
    def make(outputs=None, checkpoint: Path | None = None, tavily=None):
        deps = SimpleNamespace(
            settings=Settings(data_dir=tmp_path),
            llm=FakeLLM(outputs or _outputs()),
            tavily=tavily or FakeTavily(default_results=_search_results),
            ledger=FakeLedger(),
            memory=CompetitorMemory(tmp_path / "app.sqlite"),
        )
        runner = ResearchRunner(deps, open_checkpointer(checkpoint or tmp_path / "checkpoints.sqlite"))
        return runner, deps

    return make


APPROVE_THREE = {"approved": [{"name": "Alpha"}, {"name": "Bravo"}, {"name": "Charlie"}]}


def test_run_pauses_for_competitor_approval(make_runner):
    runner, _ = make_runner()
    status = runner.start("Us", run_id="r1")

    assert status.pending["point"] == "approve_competitors"
    assert status.pending["proposed"] == ["Alpha", "Bravo", "Charlie"]
    names = [c["name"] for c in status.pending["candidates"]]
    assert "Us" not in names and names[-1] == "Delta"  # indirect ranked last
    alpha = status.pending["candidates"][0]
    assert alpha["source_urls"] == ["https://src.example/us-competitors"]  # unknown label S9 dropped


def test_approved_competitors_are_researched_in_parallel_and_briefed(make_runner):
    runner, _ = make_runner()
    runner.start("Us", run_id="r1")
    status = runner.answer("r1", {"approved": [{"name": "Alpha"}, {"name": "Bravo"}, {"name": "New Co", "domain": "new.co"}]})

    assert status.pending["point"] == "approve_brief"
    assert status.pending["markdown"].startswith("# Brief")
    status = runner.answer("r1", {})
    assert status.done
    values = status.values
    # The company itself ("Us") is researched too, as the baseline.
    assert values["branch_status"] == {"Us": "complete", "Alpha": "complete", "Bravo": "complete", "New Co": "complete"}
    found = {c.competitor for c in values["claims"].values() if c.status == "found"}
    assert found == {"Us", "Alpha", "Bravo", "New Co"}
    assert values["approved_competitors"][2].rationale == "Added by the reviewer."
    assert [d.point for d in values["human_decisions"]] == ["approve_competitors", "approve_brief"]
    assert Path(values["brief_path"]).read_text(encoding="utf-8").startswith("# Brief")


def test_brief_feedback_rewrites_the_brief_before_export(make_runner):
    runner, deps = make_runner()
    runner.start("Us", run_id="r1")
    runner.answer("r1", APPROVE_THREE)
    status = runner.answer("r1", {"feedback": "Shorten the implications."})

    assert status.pending["point"] == "approve_brief"  # rewritten, asked again
    brief_prompts = _prompts_for(deps.llm, BriefDraft)
    assert len(brief_prompts) == 2 and "Shorten the implications." in brief_prompts[1]
    assert "brief_path" not in status.values
    assert runner.answer("r1", {}).done


def test_ambiguous_company_is_confirmed_before_discovery(make_runner):
    ambiguous = US.model_copy(update={"is_ambiguous": True, "alternatives": ["Us (us.com): docs", "Us (us.org): charity"]})
    runner, deps = make_runner(_outputs({CompanyProfile: ambiguous}))
    status = runner.start("Us", run_id="r1")
    assert status.pending["point"] == "confirm_company"
    assert not any(schema is DiscoveryPlan for schema, _ in deps.llm.calls)

    assert status.pending["reason"] == "ambiguous"

    status = runner.answer("r1", {"action": "use", "name": "Us", "domain": "us.com", "description": "The docs tool."})
    assert status.pending["point"] == "approve_competitors"
    assert status.values["company"].description == "The docs tool."
    assert status.values["company"].is_ambiguous is False


def test_junk_input_is_rejected_before_any_spend(make_runner):
    runner, deps = make_runner()
    with pytest.raises(InvalidInput):
        runner.start("#@!$%^", run_id="r1")
    with pytest.raises(InvalidInput):
        runner.start("Us", website="not a website", run_id="r1")
    assert deps.tavily.searches == [] and deps.llm.calls == []


def _prompts_for(llm, schema) -> list[str]:
    return [messages[-1][1] for s, messages in llm.calls if s is schema]


def test_market_and_region_reach_every_stage(make_runner):
    runner, deps = make_runner()
    constraints = ResearchConstraints(market="  docs   software ", region="India")
    runner.start("Us", constraints, run_id="r1")
    status = runner.answer("r1", APPROVE_THREE)

    assert status.values["constraints"].market == "docs software"  # cleaned
    for schema in (DiscoveryPlan, CandidateRanking, ResearchPlan, ClaimDrafts, ComparisonMatrix, BriefDraft):
        prompts = _prompts_for(deps.llm, schema)
        assert prompts and all("Market: docs software" in p and "Region: India" in p for p in prompts), schema


def test_missing_market_and_region_are_stated_as_unspecified(make_runner):
    runner, deps = make_runner()
    runner.start("Us", run_id="r1")
    [prompt] = _prompts_for(deps.llm, CandidateRanking)
    assert "Market: not specified" in prompt and "Region: global (not specified)" in prompt


def test_junk_region_is_rejected_before_any_spend(make_runner):
    runner, deps = make_runner()
    with pytest.raises(InvalidInput):
        runner.start("Us", ResearchConstraints(region="#@!$"), run_id="r1")
    assert deps.tavily.searches == []


def test_unknown_company_can_be_cancelled(make_runner):
    runner, deps = make_runner(_outputs({CompanyProfile: US.model_copy(update={"found": False})}))
    status = runner.start("Usz", run_id="r1")
    assert status.pending["point"] == "confirm_company"
    assert status.pending["reason"] == "not_found"

    status = runner.answer("r1", {"action": "cancel"})
    assert status.done and status.cancelled_reason == "Cancelled at company confirmation."
    assert not any(schema is DiscoveryPlan for schema, _ in deps.llm.calls)


def test_unknown_company_can_be_searched_again(make_runner):
    profiles = iter([US.model_copy(update={"found": False}), US])
    runner, deps = make_runner(_outputs({CompanyProfile: lambda messages: next(profiles)}))
    runner.start("Usz", run_id="r1")

    status = runner.answer("r1", {"action": "search_again", "name": "Us", "domain": "https://www.us.com/"})
    assert status.pending["point"] == "approve_competitors"
    scope_queries = [q for q, kw in deps.tavily.searches][:2]
    assert scope_queries == ["Usz company", "Us us.com"]


def test_website_from_user_settles_ambiguity(make_runner):
    ambiguous = US.model_copy(update={"domain": "other.com", "is_ambiguous": True, "alternatives": ["a", "b"]})
    runner, deps = make_runner(_outputs({CompanyProfile: ambiguous}))
    status = runner.start("Us", website="us.com", run_id="r1")

    assert status.pending["point"] == "approve_competitors"
    assert status.values["company"].domain == "us.com"
    assert "Website (given by the user" in deps.llm.calls[0][1][-1][1]


def test_approving_no_competitors_ends_the_run_cleanly(make_runner):
    runner, deps = make_runner()
    runner.start("Us", run_id="r1")
    status = runner.answer("r1", {"approved": []})

    assert status.done and status.cancelled_reason == "No competitors were approved."
    assert "brief" not in status.values or status.values["brief"] is None
    assert not any(schema is ResearchPlan for schema, _ in deps.llm.calls)


def test_reviewer_can_correct_a_competitor_domain(make_runner):
    runner, _ = make_runner()
    runner.start("Us", run_id="r1")
    status = runner.answer("r1", {"approved": [{"name": "Alpha", "domain": "https://www.alpha.io/x"}]})
    assert status.values["approved_competitors"][0].domain == "alpha.io"


def test_paused_run_resumes_after_restart(make_runner, tmp_path):
    first, _ = make_runner(checkpoint=tmp_path / "cp.sqlite")
    first.start("Us", run_id="r1")

    # A new process: fresh runner and checkpointer on the same database file.
    second, _ = make_runner(checkpoint=tmp_path / "cp.sqlite")
    status = second.status("r1")
    assert status.pending["point"] == "approve_competitors"
    assert isinstance(status.values["candidates"][0], Candidate)

    assert second.answer("r1", APPROVE_THREE).pending["point"] == "approve_brief"
    assert second.answer("r1", {}).done


def test_crash_after_research_resumes_without_repeating_searches(make_runner):
    crash = {"once": True}

    def flaky_synthesis(messages):
        if crash.pop("once", False):
            raise RuntimeError("process killed")
        return ComparisonMatrix(rows=[], implications=[])

    runner, deps = make_runner(_outputs({ComparisonMatrix: flaky_synthesis}))
    runner.start("Us", run_id="r1")
    with pytest.raises(RuntimeError):
        runner.answer("r1", APPROVE_THREE)

    searches_before = len(deps.tavily.searches)
    status = runner.status("r1")
    assert status.stopped and status.next_nodes == ("synthesize",)

    status = runner.continue_run("r1")
    assert status.pending["point"] == "approve_brief" and status.values["brief"] is not None
    assert len(deps.tavily.searches) == searches_before


def test_one_failed_branch_does_not_stop_the_others(make_runner):
    def plan(messages):
        if _competitor_in(messages) == "Bravo":
            raise RuntimeError("unexpected failure")
        return _research_plan(messages)

    runner, _ = make_runner(_outputs({ResearchPlan: plan}))
    runner.start("Us", run_id="r1")
    status = runner.answer("r1", APPROVE_THREE)

    assert status.values["branch_status"] == {
        "Us": "complete", "Alpha": "complete", "Bravo": "failed", "Charlie": "complete",
    }
    assert status.pending["point"] == "review_issues"
    assert status.pending["issues"] == [
        {"competitor": "Bravo", "kind": "failed", "detail": "RuntimeError: unexpected failure"}
    ]
    status = runner.answer("r1", {})  # continue without Bravo
    assert status.pending["point"] == "approve_brief" and status.values["brief"] is not None


def test_progress_is_reported_and_runs_are_listed(make_runner):
    runner, _ = make_runner()
    steps = []
    runner.start("Us", run_id="r1", on_step=steps.append)
    assert steps == [
        "scope_search", "resolve_scope", "confirm_company", "recall_competitors",
        "plan_discovery", "discovery_search", "rank_candidates",
    ]
    steps.clear()
    runner.answer("r1", APPROVE_THREE, on_step=steps.append)
    assert steps.count("research_branch") == 4 and steps[-1] == "write_brief"  # company + 3

    runner.start("Other", run_id="r2")
    assert runner.list_runs() == [("r2", "Us"), ("r1", "Us")]  # newest first; fake LLM names both "Us"


def test_next_run_reuses_approved_competitors_and_skips_discovery(make_runner):
    runner, deps = make_runner()
    runner.start("Us", run_id="r1")
    runner.answer("r1", {"approved": [{"name": "Bravo", "domain": "bravo.io"}, {"name": "New Co"}]})

    discovery_searches = len([q for q, kw in deps.tavily.searches if q == "us competitors"])
    status = runner.start("Us", run_id="r2")
    pending = status.pending
    assert pending["point"] == "approve_competitors"
    assert pending["remembered"]["run_id"] == "r1"
    assert pending["proposed"] == ["Bravo", "New Co"]
    assert pending["candidates"][0]["domain"] == "bravo.io"  # the reviewer's correction is kept
    assert len([q for q, kw in deps.tavily.searches if q == "us competitors"]) == discovery_searches

    # Asking to rediscover runs Discovery and proposes fresh candidates.
    status = runner.answer("r2", {"rediscover": True})
    assert status.pending["point"] == "approve_competitors"
    assert status.pending["remembered"] is None
    assert status.pending["proposed"] == ["Alpha", "Bravo", "Charlie"]


def test_remembered_competitors_depend_on_market_and_region(make_runner):
    runner, _ = make_runner()
    runner.start("Us", run_id="r1")
    runner.answer("r1", {"approved": [{"name": "Bravo"}]})
    status = runner.start("Us", ResearchConstraints(region="India"), run_id="r2")
    assert status.pending["remembered"] is None


def test_answering_a_run_that_is_not_paused_is_an_error(make_runner):
    runner, _ = make_runner()
    runner.start("Us", run_id="r1")
    runner.answer("r1", {"cancel": True})
    with pytest.raises(ValueError, match="isn't waiting"):
        runner.answer("r1", {})


def test_fan_out_sends_only_rerun_competitors():
    competitors = [
        Candidate(name=n, rank=i, is_direct=True, overlap_score=0.5, rationale="x")
        for i, n in enumerate(["Alpha", "Bravo"], start=1)
    ]
    state = {"run_id": "r1", "company": US, "approved_competitors": competitors, "constraints": ResearchConstraints()}
    sends = fan_out(state)
    assert [s.arg["competitor"].name for s in sends] == ["Us", "Alpha", "Bravo"]
    assert sends[0].arg["competitor"].is_self and not sends[1].arg["competitor"].is_self
    state["rerun_competitors"] = ["bravo"]
    assert [s.arg["competitor"].name for s in fan_out(state)] == ["Bravo"]
    state["cancelled_reason"] = "stop"
    assert fan_out(state) == END
