from datetime import datetime, timedelta, timezone

from competitor_research_agent.tools.cache import SearchCache


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 8, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def test_hit_returns_response_and_original_fetch_time(tmp_path):
    clock = Clock()
    cache = SearchCache(tmp_path / "app.sqlite", now=clock)
    cache.set("search", {"query": "acme"}, {"results": [1]}, ttl_hours=24)
    fetched = clock.now

    clock.now += timedelta(hours=3)
    hit = cache.get("search", {"query": "acme"})
    assert hit.response == {"results": [1]}
    assert hit.fetched_at == fetched


def test_expired_entry_is_a_miss(tmp_path):
    clock = Clock()
    cache = SearchCache(tmp_path / "app.sqlite", now=clock)
    cache.set("search", {"query": "acme"}, {"results": []}, ttl_hours=24)
    clock.now += timedelta(hours=25)
    assert cache.get("search", {"query": "acme"}) is None


def test_key_depends_on_all_params_not_their_order(tmp_path):
    cache = SearchCache(tmp_path / "app.sqlite")
    cache.set("search", {"query": "acme", "depth": "basic"}, {"results": [1]}, ttl_hours=1)
    assert cache.get("search", {"depth": "basic", "query": "acme"}) is not None
    assert cache.get("search", {"query": "acme", "depth": "advanced"}) is None
    assert cache.get("extract", {"query": "acme", "depth": "basic"}) is None
