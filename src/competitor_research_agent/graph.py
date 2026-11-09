"""LangGraph wiring.

Full run:
    scope_search -> resolve_scope -> confirm_company (human, if ambiguous or
    not found; may search again or cancel) -> recall_competitors (reuse the
    competitors approved for this company before, skipping Discovery) or
    plan_discovery -> discovery_search -> rank_candidates
    -> approve_competitors (human; may cancel or ask to rediscover)
    -> fan_out: one research_branch for the company (baseline) and one per
    competitor
    -> issue_gate (after all branches finish) -> review_issues (human, only if
    something is flagged; may rerun competitors, exclude claims, or cancel)
    -> synthesize -> write_brief -> approve_brief (human; feedback rewrites the
    brief) -> export_brief

Each research branch:
    plan_queries -> run_searches -> map_pricing_pages -> extract_pages -> extract_claims
    -> check_quotes -> assess_evidence -> back to plan_queries for gaps (at
    most max_branch_retries times, budget permitting), or finish
"""

import sqlite3
from pathlib import Path

import pydantic
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import schemas
from .agents.brief_writer import approve_brief, route_after_brief_review, write_brief
from .agents.common import research_subjects
from .agents.discovery import (
    approve_competitors,
    discovery_search,
    plan_discovery,
    rank_candidates,
    recall_competitors,
    route_after_recall,
)
from .agents.extraction import extract_claims
from .agents.research import extract_pages, map_pricing_pages, plan_queries, run_searches
from .agents.scope import confirm_company, resolve_scope, route_after_confirm, scope_search
from .agents.synthesizer import synthesize
from .agents.verifier import (
    assess_evidence,
    check_quotes,
    issue_gate,
    make_route_after_verifier,
    review_issues,
    route_after_gate,
)
from .deps import Deps
from .export import export_markdown
from .schemas import RunError
from .state import BranchState, ResearchState


def _bind(node, deps: Deps):
    return lambda state: node(state, deps)


def build_branch_graph(deps: Deps):
    builder = StateGraph(BranchState)
    steps = [
        plan_queries, run_searches, map_pricing_pages, extract_pages,
        extract_claims, check_quotes, assess_evidence,
    ]
    for node in steps:
        builder.add_node(node.__name__, _bind(node, deps))
    builder.add_edge(START, "plan_queries")
    for before, after in zip(steps, steps[1:]):
        builder.add_edge(before.__name__, after.__name__)
    builder.add_conditional_edges("assess_evidence", make_route_after_verifier(deps), ["plan_queries", END])
    return builder.compile()


def fan_out(state: ResearchState) -> list[Send] | str:
    """One branch for the company (the baseline) and one per approved
    competitor, or only those named for a rerun. A rerun branch starts with
    its earlier evidence and claims, so it can supersede them. Ends the run if
    the reviewer cancelled."""
    if state.get("cancelled_reason") or not state.get("approved_competitors"):
        return END
    rerun = {name.casefold() for name in state.get("rerun_competitors", [])}
    constraints = state["constraints"]
    sends = []
    for competitor in research_subjects(state):
        if rerun and competitor.name.casefold() not in rerun:
            continue
        sends.append(Send("research_branch", {
            "run_id": state["run_id"],
            "competitor": competitor,
            "news_days": constraints.news_days,
            "market": constraints.market,
            "region": constraints.region,
            "retry_count": 0,
            "evidence": {k: v for k, v in state.get("evidence", {}).items() if v.competitor == competitor.name},
            "claims": {k: v for k, v in state.get("claims", {}).items() if v.competitor == competitor.name},
        }))
    return sends


def route_after_approval(state: ResearchState) -> list[Send] | str:
    if state.get("rediscover"):
        return "recall_competitors"  # clears the flag, then runs Discovery
    return fan_out(state)


def route_after_review(state: ResearchState) -> list[Send] | str:
    if state.get("cancelled_reason"):
        return END
    if state.get("rerun_competitors"):
        return fan_out(state)
    return "synthesize"


def make_research_branch(deps: Deps):
    branch_graph = build_branch_graph(deps)

    def research_branch(branch_input: BranchState) -> dict:
        """Run one branch, catching its errors so one failed competitor
        doesn't stop the others."""
        name = branch_input["competitor"].name
        try:
            result = branch_graph.invoke(branch_input)
        except Exception as exc:
            return {
                "branch_status": {name: "failed"},
                "errors": [RunError(node="research_branch", competitor=name, message=f"{type(exc).__name__}: {exc}")],
            }
        return {
            "evidence": result.get("evidence", {}),
            "claims": result.get("claims", {}),
            "branch_status": {name: "complete"},
            "branch_gaps": {name: result.get("gaps", [])},
            "errors": result.get("errors", []),
        }

    return research_branch


def export_brief(state: ResearchState, deps: Deps) -> dict:
    brief = state.get("brief")
    if brief is None:
        return {}
    path = export_markdown(
        brief, state.get("claims", {}), state.get("evidence", {}),
        deps.settings.data_dir / "briefs", state["run_id"],
    )
    return {"brief_path": str(path)}


def build_research_graph(deps: Deps, checkpointer=None):
    builder = StateGraph(ResearchState)
    steps = [
        scope_search, resolve_scope, confirm_company, recall_competitors,
        plan_discovery, discovery_search, rank_candidates, approve_competitors,
        issue_gate, review_issues, synthesize, write_brief, approve_brief, export_brief,
    ]
    for node in steps:
        builder.add_node(node.__name__, _bind(node, deps))
    builder.add_node("research_branch", make_research_branch(deps))

    builder.add_edge(START, "scope_search")
    for before, after in [
        ("scope_search", "resolve_scope"),
        ("resolve_scope", "confirm_company"),
        ("plan_discovery", "discovery_search"),
        ("discovery_search", "rank_candidates"),
        ("rank_candidates", "approve_competitors"),
        ("research_branch", "issue_gate"),
        ("synthesize", "write_brief"),
        ("write_brief", "approve_brief"),
        ("export_brief", END),
    ]:
        builder.add_edge(before, after)
    builder.add_conditional_edges(
        "approve_brief", route_after_brief_review, ["write_brief", "export_brief"]
    )
    builder.add_conditional_edges(
        "confirm_company", route_after_confirm, ["scope_search", "recall_competitors", END]
    )
    builder.add_conditional_edges(
        "recall_competitors", route_after_recall, ["approve_competitors", "plan_discovery"]
    )
    builder.add_conditional_edges(
        "approve_competitors", route_after_approval, ["research_branch", "recall_competitors", END]
    )
    builder.add_conditional_edges("issue_gate", route_after_gate, ["review_issues", "synthesize"])
    builder.add_conditional_edges(
        "review_issues", route_after_review, ["research_branch", "synthesize", END]
    )
    return builder.compile(checkpointer=checkpointer)


def _schema_types() -> list[tuple[str, str]]:
    return [
        (obj.__module__, obj.__name__)
        for obj in vars(schemas).values()
        if isinstance(obj, type) and issubclass(obj, pydantic.BaseModel) and obj.__module__ == schemas.__name__
    ]


def open_checkpointer(db_path: Path) -> SqliteSaver:
    """SQLite checkpointer that only reloads our own schema types."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_schema_types())
    return SqliteSaver(conn, serde=serde)
