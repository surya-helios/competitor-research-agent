"""Brief Writer: turn the comparison and claims into a cited brief."""

from langgraph.types import interrupt
from pydantic import BaseModel

from ..deps import Deps
from ..export import render_markdown
from ..llm import LLMError
from ..schemas import Brief, Decision, RunError
from ..state import ResearchState, active_claims
from .common import cited_labels, describe_scope, format_claims, label_claims

SYSTEM = (
    "You write competitor analysis briefs for product and strategy teams. "
    "Write at most 1,000 words in Markdown with these sections: Company "
    "baseline (the company being analysed), Competitor snapshots (one short "
    "subsection each, noting how it differs from the company), Comparison (a "
    "table of pricing, core features, and positioning, with the company in the "
    "first row), Recent news, and Implications for the company.\n"
    "Rules:\n"
    "- Every fact must cite the claim it comes from, like [C3].\n"
    "- Use only the claims given. Never add facts, prices, or dates.\n"
    "- Where data is not_found or not_public, say so plainly.\n"
    "- If a market or region is given, state it in the executive summary as the "
    "scope of the brief.\n"
    "- List what could not be found or verified in `gaps`, not in the body. "
    "The body must not repeat the executive summary or have a Gaps section."
)


class BriefDraft(BaseModel):
    title: str
    executive_summary: str
    body_markdown: str
    gaps: list[str]


def write_brief(state: ResearchState, deps: Deps) -> dict:
    feedback = state.get("brief_feedback")
    labelled = label_claims(active_claims(state.get("claims", {})))
    company = state["company"]
    comparison = state.get("comparison")
    constraints = state.get("constraints")
    scope = describe_scope(constraints.market, constraints.region) if constraints else ""
    request = (
        f"Company being analysed (baseline): {company.name}\n{scope}\n\n"
        f"Claims:\n{format_claims(labelled, state.get('evidence', {}))}"
    )
    if comparison:
        request += f"\n\nAnalyst comparison (JSON):\n{comparison.model_dump_json(exclude={'rows': {'__all__': {'claim_ids'}}})}"
    if feedback:
        request += f"\n\nReviewer feedback to address:\n{feedback}"

    try:
        draft = deps.llm.structured(
            role="strong", schema=BriefDraft,
            messages=[("system", SYSTEM), ("human", request)],
            run_id=state["run_id"], node="write_brief", max_output_tokens=3000,
        )
    except LLMError as exc:
        return {"brief": None, "errors": [RunError(node="write_brief", message=str(exc))]}

    labels = cited_labels(draft.executive_summary, draft.body_markdown)
    brief = Brief(
        title=draft.title,
        executive_summary=draft.executive_summary,
        body_markdown=draft.body_markdown,
        gaps=draft.gaps,
        citations={label: labelled[label].id for label in labels if label in labelled},
    )
    return {"brief": brief}


def approve_brief(state: ResearchState, deps: Deps) -> dict:
    """Human checkpoint, always (when there is a brief). Resume with {} to
    approve, or {"feedback": "..."} to have the brief rewritten."""
    brief = state.get("brief")
    if brief is None:
        return {}
    answer = interrupt({
        "point": "approve_brief",
        "title": brief.title,
        "markdown": render_markdown(brief, state.get("claims", {}), state.get("evidence", {})),
    })
    feedback = (answer.get("feedback") or "").strip() or None
    return {
        "brief_feedback": feedback,
        "human_decisions": [Decision(point="approve_brief", payload=answer)],
    }


def route_after_brief_review(state: ResearchState) -> str:
    return "write_brief" if state.get("brief_feedback") else "export_brief"
