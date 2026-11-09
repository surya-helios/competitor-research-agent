"""Tavily search and extract with caching, budget checks, and retries.

Every paid call goes through here: check the cache, reserve worst-case credits
in the ledger, call Tavily with backoff on transient errors, then settle the
actual credits and cache the result. Search depth is always set explicitly so
Tavily never upgrades a request to a pricier depth on its own.
"""

import math
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal

import requests
from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError
from tavily.errors import UsageLimitExceededError

from ..budget import BudgetLedger
from ..config import Settings
from .cache import SearchCache

Depth = Literal["basic", "advanced"]
Topic = Literal["general", "news"]

SEARCH_CREDITS: dict[Depth, int] = {"basic": 1, "advanced": 2}
# Extract is billed per 5 successful URLs.
EXTRACT_CREDITS_PER_5: dict[Depth, int] = {"basic": 1, "advanced": 2}
# Map (without natural-language instructions) is billed per 10 pages found.
MAP_CREDITS_PER_10 = 1


class SearchError(Exception):
    """A Tavily call failed after retries, or failed in a way retries can't fix."""


@dataclass
class SearchResponse:
    results: list[dict[str, Any]]
    fetched_at: datetime
    from_cache: bool


@dataclass
class ExtractResponse:
    results: list[dict[str, Any]] = field(default_factory=list)
    failed_urls: list[str] = field(default_factory=list)
    cached_urls: list[str] = field(default_factory=list)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (UsageLimitExceededError, TavilyTimeoutError)):  # 429, timeout
        return True
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500
    return False


def _extract_credits(successful: int, depth: Depth) -> int:
    return math.ceil(successful / 5) * EXTRACT_CREDITS_PER_5[depth]


class TavilyTool:
    def __init__(
        self,
        settings: Settings,
        ledger: BudgetLedger,
        cache: SearchCache,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 3,
    ):
        self._settings = settings
        self._ledger = ledger
        self._cache = cache
        self._client = client or TavilyClient(api_key=settings.tavily_api_key)
        self._sleep = sleep
        self._max_attempts = max_attempts

    def _call_with_retries(self, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        for attempt in range(self._max_attempts):
            try:
                return call()
            except Exception as exc:
                if not _is_retryable(exc) or attempt == self._max_attempts - 1:
                    raise SearchError(f"{type(exc).__name__}: {exc}") from exc
                self._sleep(2**attempt + random.uniform(0, 1))
        raise AssertionError("unreachable")

    def search(
        self,
        run_id: str,
        node: str,
        query: str,
        *,
        depth: Depth = "basic",
        topic: Topic = "general",
        since_days: int | None = None,
        max_results: int = 5,
        include_domains: list[str] | None = None,
        include_raw_content: bool = False,
        ttl_hours: float | None = None,
    ) -> SearchResponse:
        params: dict[str, Any] = {
            "query": query,
            "search_depth": depth,
            "topic": topic,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "include_published_date": True,
        }
        if since_days is not None:
            params["start_date"] = (date.today() - timedelta(days=since_days)).isoformat()
        if include_domains:
            params["include_domains"] = sorted(include_domains)

        cached = self._cache.get("search", params)
        if cached:
            return SearchResponse(cached.response["results"], cached.fetched_at, True)

        entry = self._ledger.reserve(run_id, "tavily", SEARCH_CREDITS[depth], node=node)
        try:
            response = self._call_with_retries(
                lambda: self._client.search(**params, include_usage=True)
            )
        except BaseException:
            self._ledger.settle(entry, 0)
            raise
        self._ledger.settle(
            entry, response.get("usage", {}).get("credits", SEARCH_CREDITS[depth])
        )

        if ttl_hours is None:
            ttl_hours = (
                self._settings.cache_ttl_news_hours
                if topic == "news"
                else self._settings.cache_ttl_pages_hours
            )
        results = response.get("results", [])
        fetched_at = self._cache.set("search", params, {"results": results}, ttl_hours)
        return SearchResponse(results, fetched_at, False)

    def map_site(
        self,
        run_id: str,
        node: str,
        url: str,
        *,
        select_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[str]:
        """List URLs on a website, optionally only paths matching regexes.
        Used to find pages such as pricing that search results miss."""
        params: dict[str, Any] = {
            "url": url,
            "max_depth": max_depth,
            "limit": limit,
            "select_paths": select_paths,
            "exclude_paths": exclude_paths,
            "allow_external": False,
        }
        cached = self._cache.get("map", params)
        if cached:
            return cached.response["results"]

        entry = self._ledger.reserve(
            run_id, "tavily", math.ceil(limit / 10) * MAP_CREDITS_PER_10, node=node
        )
        try:
            response = self._call_with_retries(
                lambda: self._client.map(
                    **{k: v for k, v in params.items() if v is not None}, include_usage=True
                )
            )
        except BaseException:
            self._ledger.settle(entry, 0)
            raise
        results = response.get("results", [])
        self._ledger.settle(
            entry,
            response.get("usage", {}).get("credits", math.ceil(len(results) / 10) * MAP_CREDITS_PER_10),
        )
        self._cache.set("map", params, {"results": results}, self._settings.cache_ttl_pages_hours)
        return results

    def extract(
        self, run_id: str, node: str, urls: list[str], *, depth: Depth = "basic"
    ) -> ExtractResponse:
        """Fetch clean page text. Each URL is cached separately so a pricing
        page fetched in one run is reused by the next."""
        out = ExtractResponse()
        to_fetch = []
        for url in dict.fromkeys(urls):
            cached = self._cache.get("extract", {"url": url, "depth": depth})
            if cached:
                out.results.append({**cached.response, "fetched_at": cached.fetched_at})
                out.cached_urls.append(url)
            else:
                to_fetch.append(url)
        if not to_fetch:
            return out

        entry = self._ledger.reserve(
            run_id, "tavily", _extract_credits(len(to_fetch), depth), node=node
        )
        try:
            response = self._call_with_retries(
                lambda: self._client.extract(
                    urls=to_fetch, extract_depth=depth, format="text", include_usage=True
                )
            )
        except BaseException:
            self._ledger.settle(entry, 0)
            raise
        results = response.get("results", [])
        self._ledger.settle(
            entry,
            response.get("usage", {}).get("credits", _extract_credits(len(results), depth)),
        )

        for result in results:
            page = {"url": result["url"], "raw_content": result.get("raw_content", "")}
            fetched_at = self._cache.set(
                "extract",
                {"url": result["url"], "depth": depth},
                page,
                self._settings.cache_ttl_pages_hours,
            )
            out.results.append({**page, "fetched_at": fetched_at})
        out.failed_urls.extend(failed["url"] for failed in response.get("failed_results", []))
        return out
