"""Graph state.

ResearchState is the shared run record. Parallel research branches each get a
private BranchState and return their results, which the reducers below merge
into the shared state. Spend is tracked in the SQLite budget ledger, not here,
because parallel branches can't see each other's state until they merge.
"""

import operator
from typing import Annotated, TypedDict

from .schemas import (
    Brief,
    BranchStatus,
    Candidate,
    Claim,
    CompanyProfile,
    ComparisonMatrix,
    Decision,
    Evidence,
    PlannedQuery,
    ResearchConstraints,
    RunError,
)


def merge_dicts[V](left: dict[str, V] | None, right: dict[str, V] | None) -> dict[str, V]:
    """Merge by key; the newer value wins. Lets branches add results without
    overwriting each other, and lets a rerun replace a claim with the same ID."""
    return {**(left or {}), **(right or {})}


class ResearchState(TypedDict, total=False):
    run_id: str
    company: CompanyProfile
    constraints: ResearchConstraints

    # Search results used by Scope and Discovery: url, title, content.
    scope_sources: list[dict]
    discovery_queries: list[str]
    discovery_sources: list[dict]
    candidates: list[Candidate]
    # Set when candidates come from an earlier approval: run_id, approved_at.
    remembered: dict | None
    # Set when the reviewer asks to run Discovery instead of using remembered competitors.
    rediscover: bool
    approved_competitors: list[Candidate]
    # Competitor names to send on a rerun; empty means all approved competitors.
    rerun_competitors: list[str]

    evidence: Annotated[dict[str, Evidence], merge_dicts]
    claims: Annotated[dict[str, Claim], merge_dicts]
    branch_status: Annotated[dict[str, BranchStatus], merge_dicts]
    # Gaps the Verifier still saw when each branch finished.
    branch_gaps: Annotated[dict[str, list[str]], merge_dicts]

    errors: Annotated[list[RunError], operator.add]
    human_decisions: Annotated[list[Decision], operator.add]

    comparison: ComparisonMatrix | None
    brief: Brief | None
    # Reviewer's requested changes; the brief is rewritten while this is set.
    brief_feedback: str | None
    brief_path: str | None

    # Problems found after research, for the reviewer: competitor, kind, detail.
    open_issues: list[dict]

    # Set when the reviewer asks to search for the company again.
    rescope: bool
    # Set when the run ends early, e.g. the reviewer cancels.
    cancelled_reason: str | None


class BranchState(TypedDict, total=False):
    run_id: str
    competitor: Candidate
    news_days: int
    market: str | None
    region: str | None
    queries: list[PlannedQuery]
    pages_to_extract: list[str]
    evidence: Annotated[dict[str, Evidence], merge_dicts]
    claims: Annotated[dict[str, Claim], merge_dicts]
    retry_count: int
    gaps: list[str]
    errors: Annotated[list[RunError], operator.add]


def active_claims(claims: dict[str, Claim]) -> list[Claim]:
    """Claims that may be used in synthesis and the brief."""
    return [c for c in claims.values() if not c.superseded and not c.excluded_reason]
