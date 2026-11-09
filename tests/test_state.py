from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from competitor_research_agent.schemas import Claim, make_claim_id
from competitor_research_agent.state import ResearchState, active_claims, merge_dicts


def _claim(competitor: str, value: str, **kwargs) -> Claim:
    return Claim(
        id=make_claim_id(competitor, "pricing"),
        competitor=competitor,
        field="pricing",
        status="found",
        value=value,
        evidence_ids=["ev-1"],
        **kwargs,
    )


def test_merge_dicts_handles_empty_sides():
    assert merge_dicts(None, {"a": 1}) == {"a": 1}
    assert merge_dicts({"a": 1}, None) == {"a": 1}


def test_rerun_replaces_claim_with_same_id():
    old = {_claim("Acme", "$10").id: _claim("Acme", "$10")}
    new = {_claim("Acme", "$12").id: _claim("Acme", "$12")}
    merged = merge_dicts(old, new)
    assert len(merged) == 1
    assert next(iter(merged.values())).value == "$12"


def test_active_claims_excludes_superseded():
    claims = {
        "a": _claim("Acme", "$10", superseded=True),
        "b": _claim("Beta", "$20"),
    }
    assert [c.competitor for c in active_claims(claims)] == ["Beta"]


class _BranchInput(TypedDict):
    competitor: str


def test_parallel_branches_merge_without_overwriting():
    """Three Send branches write to the same keys in one step; reducers must
    combine them rather than raise or keep only the last."""

    def fan_out(state: ResearchState):
        return [Send("branch", {"competitor": name}) for name in ("Acme", "Beta", "Gamma")]

    def branch(branch_input: _BranchInput):
        name = branch_input["competitor"]
        claim = _claim(name, "$1")
        return {"claims": {claim.id: claim}, "branch_status": {name: "complete"}}

    builder = StateGraph(ResearchState)
    builder.add_node("start", lambda state: {})
    builder.add_node("branch", branch)
    builder.add_edge(START, "start")
    builder.add_conditional_edges("start", fan_out, ["branch"])
    builder.add_edge("branch", END)

    result = builder.compile().invoke({"run_id": "run-1"})

    assert len(result["claims"]) == 3
    assert result["branch_status"] == {
        "Acme": "complete",
        "Beta": "complete",
        "Gamma": "complete",
    }
