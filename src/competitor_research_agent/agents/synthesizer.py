"""Comparative Synthesizer: compare competitors using verified claims only."""

from ..deps import Deps
from ..llm import LLMError
from ..schemas import ComparisonMatrix, RunError
from ..state import ResearchState, active_claims
from .common import describe_scope, format_claims, label_claims

SYSTEM = (
    "You are a market research analyst comparing competitors. Use only the "
    "claims given. In each row, list the labels (e.g. C3) of the claims it "
    "relies on in claim_ids. Where a claim's status is not_found or not_public, "
    "say so instead of guessing. Claims about the company being analysed are "
    "the baseline: make its row first, and compare each competitor against it. "
    "Implications are short 'so what' points for the company being analysed, "
    "within the given market and region."
)


def synthesize(state: ResearchState, deps: Deps) -> dict:
    labelled = label_claims(active_claims(state.get("claims", {})))
    company = state["company"]
    constraints = state.get("constraints")
    scope = describe_scope(constraints.market, constraints.region) if constraints else ""
    try:
        matrix = deps.llm.structured(
            role="strong", schema=ComparisonMatrix,
            messages=[
                ("system", SYSTEM),
                (
                    "human",
                    f"Company being analysed (baseline): {company.name}"
                    + (f" ({company.description})" if company.description else "")
                    + f"\n{scope}"
                    + f"\n\nClaims:\n{format_claims(labelled, state.get('evidence', {}))}",
                ),
            ],
            run_id=state["run_id"], node="synthesize", max_output_tokens=3000,
        )
    except LLMError as exc:
        return {"comparison": None, "errors": [RunError(node="synthesize", message=str(exc))]}

    for row in matrix.rows:
        row.claim_ids = [labelled[label].id for label in row.claim_ids if label in labelled]
    return {"comparison": matrix}
