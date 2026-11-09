from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from fakes import FakeLedger, FakeLLM
from langgraph.graph import END

from competitor_research_agent.agents.common import label_claims
from competitor_research_agent.agents.extraction import extract_claims
from competitor_research_agent.agents.verifier import (
    REVIEWER_EXCLUDED,
    assess_evidence,
    check_quotes,
    issue_gate,
    make_route_after_verifier,
    names_company,
    quote_match,
    review_issues,
)
from competitor_research_agent.config import Settings
from competitor_research_agent.schemas import (
    Candidate,
    Claim,
    ClaimDraft,
    ClaimDrafts,
    CompanyProfile,
    Evidence,
    EvidenceVerdict,
    Gap,
    make_claim_id,
)
from competitor_research_agent.state import active_claims

PLANE = Candidate(name="Plane", domain="plane.so", rank=1, is_direct=True, overlap_score=0.9, rationale="x")
TODAY = date.today()


def _evidence(eid, excerpt, source_type="third_party", url="https://review.example/plane", title=None):
    return Evidence(id=eid, competitor="Plane", url=url, title=title, excerpt=excerpt, source_type=source_type)


def _claim(field, item, quote, eid="ev-1", published_at=None, value="v"):
    return Claim(
        id=make_claim_id("Plane", field, item if field != "positioning" else None),
        competitor="Plane", field=field, item=item, value=value, status="found",
        evidence_ids=[eid], quote=quote, published_at=published_at,
    )


def _check(claims, evidence):
    state = {"competitor": PLANE, "news_days": 90, "evidence": evidence, "claims": {c.id: c for c in claims}}
    return check_quotes(state, None)["claims"]


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("Pro: $6 per seat/month", "exact"),
        ("pro: $6 PER seat/month.", "exact"),  # case and trailing punctuation
        ("Pro: $6 per seat / month billed annually for teams", "exact"),  # spacing
        ("Plane pricing. Pro: $6 per seat/month billed annually for teams of all", "near"),
        ("Enterprise: $99 one time", "missing"),
    ],
)
def test_quote_match(quote, expected):
    source = "Plane pricing. **Pro**: $6 per seat/month billed annually for teams of any size."
    assert quote_match(quote, source) == expected


def test_names_company_needs_whole_word():
    assert names_company("Plane", "Plane is an open-source tool")
    assert not names_company("Plane", "Planes: interactive components")
    assert names_company("GitHub Issues", "GitHub adds sub-issues")


def test_verified_first_party_claim_gets_high_confidence():
    evidence = {"ev-1": _evidence("ev-1", "Pro: $6 per seat/month", source_type="first_party", url="https://plane.so/pricing")}
    [claim] = _check([_claim("pricing", "Pro", "Pro: $6 per seat/month")], evidence).values()
    assert claim.excluded_reason is None and claim.confidence == "high"
    assert claim.checks == ["quote verified"]


def test_claims_failing_checks_are_excluded_with_a_reason():
    old = TODAY - timedelta(days=400)
    evidence = {
        "ev-1": _evidence("ev-1", "Plane raised a Series A. Pro costs $6."),
        "ev-2": _evidence("ev-2", "Planes: $99 one time", url="https://useplanes.com"),
    }
    claims = [
        _claim("feature", "AI", "Plane has an AI agent", eid="ev-1"),  # not in source
        _claim("pricing", "Lifetime", "$99 one time", eid="ev-2"),  # different company
        _claim("news", "Series A", "Plane raised a Series A", eid="ev-1", published_at=old),  # stale
        _claim("pricing", "Pro", "Pro costs $6", eid="ev-1", published_at=old),  # old third-party price
        _claim("positioning", "p", "Plane raised", eid="ev-missing"),  # no source
    ]
    reasons = {c.field + ":" + c.id.split("-")[-1]: c.excluded_reason for c in _check(claims, evidence).values()}
    assert sorted(reasons.values()) == sorted([
        "quote not found in source",
        "source doesn't mention Plane",
        f"news older than 90 days ({old})",
        f"third-party pricing from {old} may be outdated",
        "source missing",
    ])


def test_old_first_party_price_is_kept_but_flagged():
    old = TODAY - timedelta(days=900)
    evidence = {"ev-1": _evidence("ev-1", "Suite: $11.95 per month", source_type="first_party",
                                  url="https://plane.so/press")}
    [claim] = _check([_claim("pricing", "Suite", "Suite: $11.95 per month", published_at=old)], evidence).values()
    assert claim.excluded_reason is None and claim.confidence == "low"
    assert f"source dated {old}; price may be outdated" in claim.checks


def test_badly_extracted_price_is_kept_at_low_confidence():
    # Animated price widgets extract as "$66" for a $6 price.
    evidence = {"ev-1": _evidence("ev-1", "Pro save 25% $66 per seat per month", source_type="first_party",
                                  url="https://plane.so/pricing")}
    price = _claim("pricing", "Pro", "Pro • $6 per seat per month", value="$6 per seat/month")
    invented = _claim("pricing", "Team", "Team • $9 per seat", value="$9 per seat/month")
    feature = _claim("feature", "Seats", "6 seats included", value="6 seats")
    result = {c.item: c for c in _check([price, invented, feature], evidence).values()}

    assert result["Pro"].excluded_reason is None and result["Pro"].confidence == "low"
    assert result["Team"].excluded_reason == "quote not found in source"
    assert result["Seats"].excluded_reason == "quote not found in source"  # only prices get this leeway


def test_undated_news_is_kept_with_low_confidence():
    evidence = {"ev-1": _evidence("ev-1", "Plane launched AI agents.")}
    [claim] = _check([_claim("news", "AI", "Plane launched AI agents")], evidence).values()
    assert claim.excluded_reason is None and claim.confidence == "low" and "undated news" in claim.checks


def _branch_state(claims, **extra):
    return {"run_id": "r", "competitor": PLANE, "claims": {c.id: c for c in claims}, "evidence": {}, **extra}


def test_assess_marks_conflicts_excludes_wrong_company_and_reports_gaps():
    claims = [_claim("pricing", "Pro", "q1", value="$6"), _claim("pricing", "Pro annual", "q2", value="$8"),
              _claim("feature", "Lifetime", "q3")]
    labelled = label_claims(claims)
    labels = {c.value: label for label, c in labelled.items()}
    verdict = EvidenceVerdict(
        gaps=[Gap(field="pricing", reason="Business plan price")],
        conflicting_labels=[labels["$6"], labels["$8"]],
        wrong_company_labels=[next(l for l, c in labelled.items() if c.field == "feature")],
    )
    update = assess_evidence(_branch_state(claims), SimpleNamespace(llm=FakeLLM({EvidenceVerdict: verdict})))

    statuses = {c.value: c.status for c in update["claims"].values() if c.status == "conflicting"}
    assert statuses == {"$6": "conflicting", "$8": "conflicting"}
    assert any(c.excluded_reason == "about a different company" for c in update["claims"].values())
    # Code adds gaps for required fields with no usable claim, plus the model's gaps.
    assert update["gaps"] == ["positioning: no usable claim found", "pricing: Business plan price"]


def test_retry_route_respects_gaps_retry_limit_and_budget():
    deps = SimpleNamespace(settings=Settings(max_branch_retries=2), ledger=FakeLedger())
    route = make_route_after_verifier(deps)
    assert route({"run_id": "r", "gaps": []}) == END
    assert route({"run_id": "r", "gaps": ["pricing"], "retry_count": 1}) == "plan_queries"
    assert route({"run_id": "r", "gaps": ["pricing"], "retry_count": 2}) == END
    poor = SimpleNamespace(settings=Settings(), ledger=FakeLedger(remaining=3))
    assert make_route_after_verifier(poor)({"run_id": "r", "gaps": ["pricing"], "retry_count": 0}) == END


def test_reextraction_supersedes_old_claims_but_keeps_reviewer_exclusions():
    kept = _claim("pricing", "Pro", "Pro: $6").model_copy(update={"excluded_reason": REVIEWER_EXCLUDED})
    dropped = _claim("feature", "Old feature", "old")
    evidence = {"ev-1": _evidence("ev-1", "Pro: $6")}
    drafts = ClaimDrafts(claims=[
        ClaimDraft(field="pricing", item="Pro", value="$6", status="found", quote="Pro: $6", evidence_id="ev-1"),
    ])
    state = _branch_state([kept, dropped], evidence=evidence)
    update = extract_claims(state, SimpleNamespace(llm=FakeLLM({ClaimDrafts: drafts})))

    claims = update["claims"]
    assert claims[kept.id].excluded_reason == REVIEWER_EXCLUDED
    assert claims[dropped.id].superseded is True
    assert all(c.field != "feature" or c.status == "not_found" for c in active_claims(claims))


def test_issue_gate_flags_failures_missing_fields_and_conflicts():
    conflict = _claim("pricing", "Pro", "q").model_copy(update={"status": "conflicting"})
    others = [_claim("feature", "f", "q"), _claim("positioning", "p", "q")]
    state = {
        # Plane is the company being analysed (the baseline); Jira is a competitor.
        "company": CompanyProfile(name="Plane", domain="plane.so"),
        "approved_competitors": [PLANE.model_copy(update={"name": "Jira"})],
        "branch_status": {"Plane": "complete", "Jira": "failed"},
        "claims": {c.id: c for c in [conflict, *others]},
        "errors": [],
    }
    kinds = [(i["competitor"], i["kind"]) for i in issue_gate(state, None)["open_issues"]]
    # A conflicting price still counts as missing usable pricing.
    assert kinds == [("Plane", "missing"), ("Plane", "conflict"), ("Jira", "failed")]


def test_review_issues_applies_exclusions_and_reruns(monkeypatch):
    claim = _claim("pricing", "Pro", "q")
    monkeypatch.setattr(
        "competitor_research_agent.agents.verifier.interrupt",
        lambda payload: {"exclude_claims": [claim.id], "rerun": ["Plane"]},
    )
    update = review_issues({"claims": {claim.id: claim}, "open_issues": [{"x": 1}]}, None)
    assert update["claims"][claim.id].excluded_reason == REVIEWER_EXCLUDED
    assert update["rerun_competitors"] == ["Plane"] and update["open_issues"] == []
