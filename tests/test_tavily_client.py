from datetime import date, timedelta

import pytest
from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError

from competitor_research_agent.budget import BudgetExceeded, BudgetLedger
from competitor_research_agent.config import Settings
from competitor_research_agent.tools.cache import SearchCache
from competitor_research_agent.tools.tavily_client import SearchError, TavilyTool


class FakeTavily:
    def __init__(self, search_outcomes=(), extract_outcomes=()):
        self.search_outcomes = list(search_outcomes)
        self.extract_outcomes = list(extract_outcomes)
        self.search_calls = []
        self.extract_calls = []

    @staticmethod
    def _next(outcomes):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self._next(self.search_outcomes)

    def extract(self, **kwargs):
        self.extract_calls.append(kwargs)
        return self._next(self.extract_outcomes)


def _search_ok(credits=1):
    return {"results": [{"url": "https://acme.com", "content": "Acme"}], "usage": {"credits": credits}}


@pytest.fixture
def make_tool(tmp_path):
    ledger = BudgetLedger(tmp_path / "app.sqlite", {"tavily": 10, "nebius": 1})
    cache = SearchCache(tmp_path / "app.sqlite")
    sleeps = []

    def make(client):
        tool = TavilyTool(Settings(), ledger, cache, client=client, sleep=sleeps.append)
        return tool, ledger, sleeps

    return make


def test_search_is_cached_and_charged_once(make_tool):
    client = FakeTavily([_search_ok()])
    tool, ledger, _ = make_tool(client)

    first = tool.search("run-1", "discovery_search", "acme alternatives")
    second = tool.search("run-1", "discovery_search", "acme alternatives")

    assert len(client.search_calls) == 1
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.fetched_at == first.fetched_at
    assert ledger.spent("run-1", "tavily") == 1


def test_search_sets_depth_explicitly_and_requests_usage(make_tool):
    client = FakeTavily([_search_ok(2)])
    tool, ledger, _ = make_tool(client)
    tool.search("run-1", "run_searches", "acme pricing", depth="advanced")
    call = client.search_calls[0]
    assert call["search_depth"] == "advanced" and call["include_usage"] is True
    assert ledger.spent("run-1", "tavily") == 2


def test_news_window_becomes_start_date(make_tool):
    client = FakeTavily([_search_ok()])
    tool, _, _ = make_tool(client)
    tool.search("run-1", "run_searches", "acme news", topic="news", since_days=90)
    expected = (date.today() - timedelta(days=90)).isoformat()
    assert client.search_calls[0]["start_date"] == expected


def test_rate_limit_is_retried_with_backoff(make_tool):
    client = FakeTavily([UsageLimitExceededError("slow down"), _search_ok()])
    tool, ledger, sleeps = make_tool(client)
    result = tool.search("run-1", "run_searches", "acme")
    assert result.results and len(client.search_calls) == 2
    assert len(sleeps) == 1
    assert ledger.spent("run-1", "tavily") == 1


def test_bad_key_fails_fast_and_is_not_charged(make_tool):
    client = FakeTavily([InvalidAPIKeyError("bad key")])
    tool, ledger, sleeps = make_tool(client)
    with pytest.raises(SearchError):
        tool.search("run-1", "run_searches", "acme")
    assert sleeps == [] and ledger.spent("run-1", "tavily") == 0


def test_persistent_failure_raises_after_max_attempts(make_tool):
    client = FakeTavily([UsageLimitExceededError("x")] * 3)
    tool, ledger, _ = make_tool(client)
    with pytest.raises(SearchError):
        tool.search("run-1", "run_searches", "acme")
    assert len(client.search_calls) == 3
    assert ledger.spent("run-1", "tavily") == 0


def test_budget_exceeded_blocks_call(make_tool):
    client = FakeTavily([_search_ok()])
    tool, ledger, _ = make_tool(client)
    ledger.reserve("run-1", "tavily", 10, node="earlier")
    with pytest.raises(BudgetExceeded):
        tool.search("run-1", "run_searches", "acme")
    assert client.search_calls == []


def test_extract_caches_per_url_and_reports_failures(make_tool):
    client = FakeTavily(
        extract_outcomes=[
            {
                "results": [{"url": "https://a.com", "raw_content": "A page"}],
                "failed_results": [{"url": "https://b.com", "error": "blocked"}],
                "usage": {"credits": 1},
            },
            {"results": [{"url": "https://c.com", "raw_content": "C page"}], "usage": {"credits": 1}},
        ]
    )
    tool, ledger, _ = make_tool(client)

    first = tool.extract("run-1", "extract_pages", ["https://a.com", "https://b.com"])
    assert first.failed_urls == ["https://b.com"]
    assert [r["url"] for r in first.results] == ["https://a.com"]

    second = tool.extract("run-1", "extract_pages", ["https://a.com", "https://c.com"])
    assert second.cached_urls == ["https://a.com"]
    assert client.extract_calls[1]["urls"] == ["https://c.com"]
    assert ledger.spent("run-1", "tavily") == 2
