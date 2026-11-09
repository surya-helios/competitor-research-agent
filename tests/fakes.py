"""Offline stand-ins for the LLM and Tavily clients."""

from datetime import datetime, timezone

from competitor_research_agent.tools.tavily_client import ExtractResponse, SearchError, SearchResponse

NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)


class FakeLLM:
    """Returns a canned output per schema. An output may be an exception to
    raise, or a function of the messages."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def structured(self, *, schema, messages, **kwargs):
        self.calls.append((schema, messages))
        output = self.outputs[schema]
        if isinstance(output, Exception):
            raise output
        return output(messages) if callable(output) else output


class FakeLedger:
    def __init__(self, remaining=1000.0):
        self._remaining = remaining

    def remaining(self, run_id, provider):
        return self._remaining


class FakeTavily:
    """`default_results` may be a list, or a function of the query."""

    def __init__(self, search_results=None, search_errors=(), default_results=None, site_maps=None):
        self.search_results = search_results or {}
        self.search_errors = set(search_errors)
        self.default_results = default_results or []
        self.site_maps = site_maps or {}
        self.searches = []
        self.maps = []
        self.extracted = []

    def search(self, run_id, node, query, **kwargs):
        self.searches.append((query, kwargs))
        if query in self.search_errors:
            raise SearchError("boom")
        if query in self.search_results:
            return SearchResponse(self.search_results[query], NOW, False)
        default = self.default_results
        return SearchResponse(default(query) if callable(default) else default, NOW, False)

    def map_site(self, run_id, node, url, **kwargs):
        self.maps.append(url)
        return self.site_maps.get(url, [])

    def extract(self, run_id, node, urls, **kwargs):
        self.extracted.append(urls)
        return ExtractResponse(
            results=[{"url": u, "raw_content": f"Page {u}: Pro $10/user/month", "fetched_at": NOW} for u in urls],
        )
