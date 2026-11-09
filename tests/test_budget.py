import threading

import pytest

from competitor_research_agent.budget import BudgetExceeded, BudgetLedger, llm_cost_usd


@pytest.fixture
def ledger(tmp_path):
    return BudgetLedger(tmp_path / "app.sqlite", {"tavily": 10, "nebius": 0.5})


def test_reserve_counts_against_limit(ledger):
    ledger.reserve("run-1", "tavily", 4, node="discovery_search")
    assert ledger.spent("run-1", "tavily") == 4
    assert ledger.remaining("run-1", "tavily") == 6


def test_reserve_over_limit_raises_and_records_nothing(ledger):
    ledger.reserve("run-1", "tavily", 8, node="run_searches")
    with pytest.raises(BudgetExceeded) as exc:
        ledger.reserve("run-1", "tavily", 3, node="run_searches")
    assert exc.value.remaining == 2
    assert ledger.spent("run-1", "tavily") == 8


def test_settle_replaces_reservation_with_actual(ledger):
    entry = ledger.reserve("run-1", "tavily", 2, node="run_searches")
    ledger.settle(entry, 1)
    assert ledger.spent("run-1", "tavily") == 1


def test_failed_call_settled_at_zero_frees_budget(ledger):
    entry = ledger.reserve("run-1", "tavily", 10, node="extract_pages")
    ledger.settle(entry, 0)
    assert ledger.remaining("run-1", "tavily") == 10


def test_runs_and_providers_are_tracked_separately(ledger):
    ledger.reserve("run-1", "tavily", 10, node="run_searches")
    ledger.reserve("run-1", "nebius", 0.2, node="write_brief", model="m")
    assert ledger.remaining("run-2", "tavily") == 10
    assert ledger.remaining("run-1", "nebius") == pytest.approx(0.3)


def test_parallel_reservations_never_exceed_limit(ledger):
    accepted = []

    def branch():
        for _ in range(5):
            try:
                ledger.reserve("run-1", "tavily", 1, node="run_searches")
                accepted.append(1)
            except BudgetExceeded:
                pass

    threads = [threading.Thread(target=branch) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(accepted) == 10
    assert ledger.spent("run-1", "tavily") == 10


def test_spend_persists_across_ledger_instances(tmp_path):
    db = tmp_path / "app.sqlite"
    BudgetLedger(db, {"tavily": 10, "nebius": 1}).reserve("run-1", "tavily", 3, node="x")
    assert BudgetLedger(db, {"tavily": 10, "nebius": 1}).spent("run-1", "tavily") == 3


def test_summary_groups_by_node_and_model(ledger):
    first = ledger.reserve("run-1", "nebius", 0.1, node="extract_claims", model="fast")
    ledger.settle(first, 0.02, input_tokens=1000, output_tokens=200)
    ledger.reserve("run-1", "nebius", 0.1, node="extract_claims", model="fast")

    [row] = ledger.summary("run-1")
    assert (row.node, row.model, row.calls) == ("extract_claims", "fast", 2)
    assert row.amount == pytest.approx(0.12)


def test_llm_cost_usd():
    assert llm_cost_usd(1_000_000, 500_000, 0.2, 0.6) == pytest.approx(0.5)
