"""Agent 2: gather evidence for one competitor.

plan_queries (reasoning) -> run_searches (action) -> map_pricing_pages (action)
-> extract_pages (action)
"""

import re
from datetime import date
from urllib.parse import urlparse

from ..budget import BudgetExceeded
from ..deps import Deps
from ..llm import LLMError
from ..schemas import Candidate, Evidence, PlannedQuery, ResearchPlan, RunError, make_evidence_id
from ..state import BranchState
from ..tools.tavily_client import Depth, SearchError
from .common import describe_scope, is_first_party, parse_published

MAX_QUERIES = 4
# A retry searches only for the gaps the Verifier found.
MAX_RETRY_QUERIES = 2
# Full pages fetched per competitor: up to 2 for pricing, 1 for features.
MAX_PAGES = {"pricing": 2, "feature": 1}
# Pricing pages found by mapping the competitor's site, extracted ahead of the rest.
MAX_MAPPED_PRICING_PAGES = 2
MAX_PAGES_TOTAL = 4
PRICING_PATHS = [r"/.*(pric|plan|edition|cost|subscri|quote).*"]
SKIP_PATHS = [r"/(blog|news|press|community|careers?|legal|docs?)/.*"]
PAGE_CHARS = 8000
# First-party pricing and feature pages are worth the pricier search depth.
DEPTH_BY_PURPOSE: dict[str, Depth] = {
    "pricing": "advanced",
    "feature": "advanced",
    "positioning": "basic",
    "news": "basic",
}


def default_plan(competitor: Candidate, region: str | None = None) -> list[PlannedQuery]:
    """Used when the LLM can't plan, so research still runs."""
    name, own_site = competitor.name, competitor.domain is not None
    in_region = f" {region}" if region else ""
    return [
        PlannedQuery(query=f"{name} pricing plans{in_region}", purpose="pricing", use_competitor_domain=own_site),
        PlannedQuery(query=f"{name} features", purpose="feature", use_competitor_domain=own_site),
        PlannedQuery(query=f"{name} review vs alternatives", purpose="positioning", use_competitor_domain=False),
        PlannedQuery(query=f"{name} news{in_region}", purpose="news", use_competitor_domain=False),
    ]


def plan_queries(state: BranchState, deps: Deps) -> dict:
    competitor = state["competitor"]
    gaps = state.get("gaps", [])
    limit = MAX_RETRY_QUERIES if gaps else MAX_QUERIES
    task = (
        f"Fill these gaps from the previous attempt, using different searches: {'; '.join(gaps)}"
        if gaps
        else "Cover pricing, core features, market positioning, and recent news."
    )
    messages = [
        (
            "system",
            f"You plan web searches for competitor research. Today is {date.today()}. "
            f"Write at most {limit} short, specific search queries.\n"
            "- Set use_competitor_domain to true for pricing and feature queries when "
            "the competitor's website is known, so results come from first-party pages.\n"
            "- Other companies may share the competitor's name, so include its website "
            "or product category in news and positioning queries.\n"
            "- If a region is given, include it in the pricing and news queries, since "
            "prices and news can differ by region.\n"
            "- Don't put years in queries; news searches are already limited to recent dates.",
        ),
        (
            "human",
            f"Competitor: {competitor.name}\n"
            f"Website: {competitor.domain or 'unknown'}\n"
            f"{describe_scope(state.get('market'), state.get('region'))}\n"
            f"Task: {task}",
        ),
    ]
    try:
        plan = deps.llm.structured(
            role="fast", schema=ResearchPlan, messages=messages,
            run_id=state["run_id"], node="plan_queries", max_output_tokens=800,
        )
        queries = plan.queries[:limit]
        errors = []
    except LLMError as exc:
        queries, errors = [], [RunError(node="plan_queries", competitor=competitor.name, message=str(exc))]
    if not queries and not gaps:
        queries = default_plan(competitor, state.get("region"))
    return {
        "queries": queries,
        "retry_count": state.get("retry_count", 0) + (1 if gaps else 0),
        "gaps": [],
        "errors": errors,
    }


def _to_evidence(
    competitor: Candidate, result: dict, query: PlannedQuery, fetched_at
) -> Evidence:
    url = result["url"]
    excerpt = result.get("content") or ""
    if is_first_party(url, competitor.domain):
        source_type = "first_party"
    elif query.purpose == "news":
        source_type = "news"
    else:
        source_type = "third_party"
    return Evidence(
        id=make_evidence_id(competitor.name, url, excerpt),
        competitor=competitor.name,
        url=url,
        title=result.get("title"),
        excerpt=excerpt,
        source_type=source_type,
        query=query.query,
        retrieved_at=fetched_at,
        published_at=parse_published(result.get("published_date")),
    )


def _names_competitor(query: str, competitor: Candidate) -> bool:
    """True if the query names the competitor as a word. A domain alone, like
    'useplane.com', doesn't count: it may be wrong."""
    text = query.casefold()
    if competitor.domain:
        text = text.replace(competitor.domain.casefold(), " ")
    return re.search(rf"\b{re.escape(competitor.name.casefold())}\b", text) is not None


def run_searches(state: BranchState, deps: Deps) -> dict:
    competitor = state["competitor"]
    domain = competitor.domain
    evidence: dict[str, Evidence] = {}
    # Search often misses the main pricing page, so try the usual URL too.
    # A failed extraction costs nothing.
    pages: dict[str, list[str]] = {
        "pricing": [f"https://{domain}/pricing"] if domain else [],
        "feature": [],
    }
    errors: list[RunError] = []

    for query in state["queries"]:
        is_news = query.purpose == "news"
        # Rules enforced in code, whatever the plan says: every query names the
        # competitor, and only pricing and feature searches stay on its own site.
        if not _names_competitor(query.query, competitor):
            query = query.model_copy(update={"query": f"{competitor.name} {query.query}"})
        site_only = bool(domain and query.use_competitor_domain and query.purpose in MAX_PAGES)

        def search(include_domains):
            return deps.tavily.search(
                state["run_id"], "run_searches", query.query,
                depth=DEPTH_BY_PURPOSE[query.purpose],
                topic="news" if is_news else "general",
                since_days=state.get("news_days", 90) if is_news else None,
                include_domains=include_domains,
                max_results=5,
            )

        try:
            response = search([domain] if site_only else None)
            if site_only and not response.results:
                # Nothing on the competitor's site usually means the domain is wrong.
                errors.append(RunError(
                    node="run_searches", competitor=competitor.name,
                    message=f"no results on {domain} for '{query.query}'; searched the whole web. The domain may be wrong.",
                ))
                response = search(None)
        except SearchError as exc:
            errors.append(RunError(node="run_searches", competitor=competitor.name, message=f"{query.query}: {exc}"))
            continue
        except BudgetExceeded as exc:
            errors.append(RunError(node="run_searches", competitor=competitor.name, message=str(exc)))
            break

        for rank, result in enumerate(response.results):
            item = _to_evidence(competitor, result, query, response.fetched_at)
            evidence[item.id] = item
            # Full pages for pricing and features: first-party pages when the
            # domain is known, otherwise the top result.
            wanted = item.source_type == "first_party" if domain else rank == 0
            if query.purpose in pages and wanted and item.url not in pages[query.purpose]:
                pages[query.purpose].append(item.url)

    selected = [url for purpose, urls in pages.items() for url in urls[: MAX_PAGES[purpose]]]
    return {
        "evidence": evidence,
        "pages_to_extract": list(dict.fromkeys(selected)),
        "errors": errors,
    }


def _pricing_page_rank(url: str) -> tuple[int, int]:
    """Main pricing pages first, then plan or edition pages; quote forms last."""
    path = urlparse(url).path.lower()
    if "quote" in path:
        kind = 3
    elif "pric" in path:
        kind = 0
    elif "plan" in path or "edition" in path:
        kind = 1
    else:
        kind = 2
    return kind, len(path)


def map_pricing_pages(state: BranchState, deps: Deps) -> dict:
    """Search often misses pricing pages, so list the competitor's own site
    for pricing-like paths. Retries map again only when pricing is a gap."""
    competitor = state["competitor"]
    if not competitor.domain:
        return {}
    retrying = state.get("retry_count", 0) > 0
    if retrying and not any(q.purpose == "pricing" for q in state.get("queries", [])):
        return {}
    try:
        urls = deps.tavily.map_site(
            state["run_id"], "map_pricing_pages", f"https://{competitor.domain}",
            select_paths=PRICING_PATHS, exclude_paths=SKIP_PATHS,
        )
    except (SearchError, BudgetExceeded) as exc:
        return {"errors": [RunError(node="map_pricing_pages", competitor=competitor.name, message=str(exc))]}

    # One URL per page, ignoring "www." and trailing slashes.
    seen, mapped = set(), []
    for url in sorted(urls, key=_pricing_page_rank):
        parsed = urlparse(url)
        key = (parsed.netloc.lower().removeprefix("www."), parsed.path.rstrip("/").lower())
        if key not in seen and is_first_party(url, competitor.domain):
            seen.add(key)
            mapped.append(url)
    pages = mapped[:MAX_MAPPED_PRICING_PAGES] + state.get("pages_to_extract", [])
    return {"pages_to_extract": list(dict.fromkeys(pages))[:MAX_PAGES_TOTAL]}


def extract_pages(state: BranchState, deps: Deps) -> dict:
    competitor = state["competitor"]
    urls = state.get("pages_to_extract", [])
    if not urls:
        return {}
    try:
        # Advanced depth renders JavaScript-heavy pages, which many pricing pages are.
        response = deps.tavily.extract(state["run_id"], "extract_pages", urls, depth="advanced")
    except (SearchError, BudgetExceeded) as exc:
        return {"errors": [RunError(node="extract_pages", competitor=competitor.name, message=str(exc))]}

    evidence = {}
    for page in response.results:
        text = page["raw_content"][:PAGE_CHARS]
        if not text.strip():
            continue
        item = Evidence(
            id=make_evidence_id(competitor.name, page["url"], text),
            competitor=competitor.name,
            url=page["url"],
            excerpt=text,
            source_type="first_party" if is_first_party(page["url"], competitor.domain) else "third_party",
            retrieved_at=page["fetched_at"],
        )
        evidence[item.id] = item
    errors = [
        RunError(node="extract_pages", competitor=competitor.name, message=f"could not extract {url}")
        for url in response.failed_urls
    ]
    return {"evidence": evidence, "errors": errors}
