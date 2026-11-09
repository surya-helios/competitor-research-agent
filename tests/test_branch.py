from datetime import date
from types import SimpleNamespace

from fakes import FakeLedger, FakeLLM, FakeTavily

from competitor_research_agent.config import Settings

from competitor_research_agent.agents.brief_writer import BriefDraft, write_brief
from competitor_research_agent.agents.common import is_first_party, label_claims, parse_published
from competitor_research_agent.agents.extraction import drafts_to_claims
from competitor_research_agent.agents.research import map_pricing_pages, plan_queries, run_searches
from competitor_research_agent.export import render_markdown
from competitor_research_agent.graph import build_branch_graph
from competitor_research_agent.llm import LLMError
from competitor_research_agent.schemas import (
    Candidate,
    ClaimDraft,
    ClaimDrafts,
    CompanyProfile,
    Evidence,
    EvidenceVerdict,
    PlannedQuery,
    ResearchPlan,
)
ACME = Candidate(name="Acme", domain="acme.com", rank=1, is_direct=True, overlap_score=0.9, rationale="x")


def _deps(llm, tavily):
    return SimpleNamespace(llm=llm, tavily=tavily, settings=Settings(), ledger=FakeLedger())


NO_ISSUES = EvidenceVerdict(gaps=[], conflicting_labels=[], wrong_company_labels=[])


def _plan(*queries):
    return ResearchPlan(queries=[PlannedQuery(query=q, purpose=p, use_competitor_domain=d) for q, p, d in queries])


def test_parse_published_handles_tavily_and_iso_formats():
    assert parse_published("Fri, 18 Aug 2023 00:00:00 GMT") == date(2023, 8, 18)
    assert parse_published("2026-10-02T10:00:00Z") == date(2026, 10, 2)
    assert parse_published("not a date") is None and parse_published(None) is None


def test_is_first_party_matches_subdomains_only():
    assert is_first_party("https://www.acme.com/pricing", "acme.com")
    assert is_first_party("https://help.acme.com/x", "acme.com")
    assert not is_first_party("https://notacme.com/pricing", "acme.com")


def test_plan_queries_falls_back_to_templates_when_llm_fails():
    update = plan_queries(
        {"run_id": "r", "competitor": ACME}, _deps(FakeLLM({ResearchPlan: LLMError("down")}), None)
    )
    assert [q.purpose for q in update["queries"]] == ["pricing", "feature", "positioning", "news"]
    assert update["errors"][0].node == "plan_queries"


def test_retry_plan_includes_gaps_in_prompt():
    llm = FakeLLM({ResearchPlan: _plan(("acme pricing", "pricing", True))})
    plan_queries({"run_id": "r", "competitor": ACME, "gaps": ["pricing for Team plan"]}, _deps(llm, None))
    assert "pricing for Team plan" in llm.calls[0][1][-1][1]


def test_run_searches_keeps_going_after_a_failed_query():
    tavily = FakeTavily(
        search_results={
            "acme pricing": [
                {"url": "https://acme.com/pricing", "content": "Pro $10", "title": "Pricing"},
                {"url": "https://blog.example/acme", "content": "Acme costs $10"},
            ],
            "acme news": [{"url": "https://news.example/a", "content": "Acme raises", "published_date": "Mon, 02 Nov 2026 00:00:00 GMT"}],
        },
        search_errors={"acme features"},
    )
    queries = _plan(
        ("acme pricing", "pricing", True), ("acme features", "feature", True), ("acme news", "news", False)
    ).queries
    update = run_searches({"run_id": "r", "competitor": ACME, "queries": queries, "news_days": 90}, _deps(None, tavily))

    types = sorted(e.source_type for e in update["evidence"].values())
    assert types == ["first_party", "news", "third_party"]
    # The usual /pricing URL is tried first; it matches the search result here.
    assert update["pages_to_extract"] == ["https://acme.com/pricing"]
    assert len(update["errors"]) == 1
    pricing_kwargs, news_kwargs = tavily.searches[0][1], tavily.searches[2][1]
    assert pricing_kwargs["include_domains"] == ["acme.com"] and pricing_kwargs["depth"] == "advanced"
    assert news_kwargs["topic"] == "news" and news_kwargs["since_days"] == 90


def test_query_rules_are_enforced_and_wrong_domain_falls_back():
    tavily = FakeTavily(search_results={
        "Acme pricing acme.com": [],  # nothing on the (wrong) domain
    }, default_results=[{"url": "https://review.example/acme", "content": "Acme review"}])
    queries = _plan(("pricing acme.com", "pricing", True), ("recent news", "news", True)).queries
    update = run_searches({"run_id": "r", "competitor": ACME, "queries": queries}, _deps(None, tavily))

    sent = [(q, kw["include_domains"]) for q, kw in tavily.searches]
    assert sent == [
        ("Acme pricing acme.com", ["acme.com"]),
        ("Acme pricing acme.com", None),  # fallback to the whole web
        ("Acme recent news", None),  # news never limited to the competitor's site
    ]
    assert "domain may be wrong" in update["errors"][0].message
    assert len(update["evidence"]) == 1


def test_mapped_pricing_pages_are_extracted_first():
    tavily = FakeTavily(site_maps={"https://acme.com": [
        "https://acme.com/pricing/quote", "https://www.acme.com/plans",
        "https://acme.com/software-pricing/", "https://acme.com/software-pricing",
    ]})
    state = {"run_id": "r", "competitor": ACME, "pages_to_extract": ["https://acme.com/pricing", "https://acme.com/x"]}
    update = map_pricing_pages(state, _deps(None, tavily))
    assert update["pages_to_extract"] == [
        "https://acme.com/software-pricing", "https://www.acme.com/plans",  # mapped, best first, deduplicated
        "https://acme.com/pricing", "https://acme.com/x",
    ]


def test_map_is_skipped_without_domain_or_on_retries_without_pricing_gaps():
    tavily = FakeTavily()
    no_domain = ACME.model_copy(update={"domain": None})
    assert map_pricing_pages({"run_id": "r", "competitor": no_domain}, _deps(None, tavily)) == {}
    retry = {"run_id": "r", "competitor": ACME, "retry_count": 1,
             "queries": [PlannedQuery(query="Acme news", purpose="news", use_competitor_domain=False)]}
    assert map_pricing_pages(retry, _deps(None, tavily)) == {}
    assert tavily.maps == []


def test_pricing_url_is_tried_even_when_search_misses_it():
    tavily = FakeTavily(search_results={"acme pricing": [{"url": "https://acme.com/help/billing", "content": "Billing"}]})
    queries = _plan(("acme pricing", "pricing", True)).queries
    update = run_searches({"run_id": "r", "competitor": ACME, "queries": queries}, _deps(None, tavily))
    assert update["pages_to_extract"] == ["https://acme.com/pricing", "https://acme.com/help/billing"]


def _evidence(eid="ev-1", source_type="first_party"):
    return Evidence(id=eid, competitor="Acme", url="https://acme.com/pricing", excerpt="Pro $10", source_type=source_type)


def test_drafts_drop_unsupported_claims_and_fill_missing_fields():
    drafts = [
        ClaimDraft(field="pricing", item="Pro", value="$10/user/month", status="found", quote="Pro $10", evidence_id="ev-1"),
        ClaimDraft(field="feature", item="SSO", value="SAML SSO", status="found", quote="SSO", evidence_id="ev-missing"),
        ClaimDraft(field="pricing", item="Enterprise", value=None, status="not_public", quote=None, evidence_id=None),
    ]
    claims, errors = drafts_to_claims("Acme", drafts, {"ev-1": _evidence()})

    by_field = {}
    for claim in claims.values():
        by_field.setdefault(claim.field, []).append(claim)
    assert len(by_field["pricing"]) == 2
    assert by_field["feature"][0].status == "not_found"  # SSO dropped, gap filled
    assert {c.status for c in by_field["positioning"] + by_field["news"]} == {"not_found"}
    assert len(errors) == 1 and "SSO" in errors[0].message
    found = next(c for c in by_field["pricing"] if c.status == "found")
    assert found.evidence_ids == ["ev-1"] and found.confidence == "medium"


def test_branch_graph_end_to_end_with_fakes():
    llm = FakeLLM({
        ResearchPlan: _plan(("acme pricing", "pricing", True)),
        ClaimDrafts: ClaimDrafts(claims=[]),
        EvidenceVerdict: NO_ISSUES,
    })
    tavily = FakeTavily(search_results={"acme pricing": [{"url": "https://acme.com/pricing", "content": "Pro $10"}]})
    result = build_branch_graph(_deps(llm, tavily)).invoke({"run_id": "r", "competitor": ACME, "news_days": 90})

    # No claims found, so the Verifier reports gaps and the branch retries twice.
    assert result["retry_count"] == 2
    assert tavily.extracted[0] == ["https://acme.com/pricing"]
    assert len(result["evidence"]) == 2  # search snippet + extracted page
    page_text = next(messages for schema, messages in llm.calls if schema is ClaimDrafts)[-1][1]
    assert "Page https://acme.com/pricing" in page_text
    assert {c.status for c in result["claims"].values()} == {"not_found"}


def test_brief_citations_map_labels_to_claims_and_render_sources():
    claims, _ = drafts_to_claims(
        "Acme",
        [ClaimDraft(field="pricing", item="Pro", value="$10/user/month", status="found", quote="Pro $10", evidence_id="ev-1")],
        {"ev-1": _evidence()},
    )
    evidence = {"ev-1": _evidence()}
    label, pricing = next(
        (label, c) for label, c in label_claims(list(claims.values())).items() if c.status == "found"
    )
    llm = FakeLLM({BriefDraft: BriefDraft(
        title="Acme brief", executive_summary=f"Acme Pro costs $10 [{label}].",
        body_markdown=f"## Comparison\nPricing [C1, {label}], unknown label [C99].", gaps=["No news found"],
    )})
    state = {"run_id": "r", "company": CompanyProfile(name="Us"), "claims": claims, "evidence": evidence}
    brief = write_brief(state, _deps(llm, None))["brief"]

    assert brief.citations[label] == pricing.id
    assert "C1" in brief.citations and "C99" not in brief.citations
    markdown = render_markdown(brief, claims, evidence)
    assert "## Sources" in markdown and "https://acme.com/pricing" in markdown
    assert "No news found" in markdown
