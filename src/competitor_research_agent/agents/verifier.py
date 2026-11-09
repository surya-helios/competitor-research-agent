"""Verifier: check claims before they reach the brief.

In each research branch:
    check_quotes (action, code) -> assess_evidence (reasoning)
    -> route_after_verifier: retry the gaps, or finish
After all branches:
    issue_gate (control) -> review_issues (human, only if something is flagged)
"""

import re
import unicodedata
from datetime import date, timedelta

from langgraph.graph import END
from langgraph.types import interrupt

from ..deps import Deps
from ..llm import LLMError
from ..schemas import Claim, Decision, EvidenceVerdict, RunError
from ..state import BranchState, ResearchState, active_claims
from .common import describe_scope, format_claims, label_claims, research_subjects

# Fields every competitor should have a usable claim for. News can legitimately be empty.
REQUIRED_FIELDS = ("pricing", "feature", "positioning")
STALE_PRICING_DAYS = 365
# Don't start a retry unless a few searches' worth of credits remain.
RETRY_MIN_CREDITS = 6
REVIEWER_EXCLUDED = "Excluded by the reviewer."

_PUNCTUATION = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION).casefold()
    text = re.sub(r"[*_#`>]", "", text)  # Markdown left over from page extraction
    text = text.replace("|", " ")
    text = " ".join(text.split())
    text = re.sub(r"\s*/\s*", "/", text)  # "seat / month" == "seat/month"
    return re.sub(r"\s+([:;,.!?])", r"\1", text)  # "Pro :" == "Pro:"


def _trigrams(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def quote_match(quote: str, source_text: str) -> str:
    """'exact', 'near' (most of the quote's word sequences appear), or 'missing'."""
    q, text = normalise(quote).strip(" .,;:"), normalise(source_text)
    if q and q in text:
        return "exact"
    quote_trigrams = _trigrams(q.split())
    if quote_trigrams and len(quote_trigrams & _trigrams(text.split())) / len(quote_trigrams) >= 0.8:
        return "near"
    return "missing"


def numbers_in_source(value: str, source_text: str) -> bool:
    """True if the value has numbers and each starts a number in the source."""
    numbers = re.findall(r"\d+(?:[.,]\d+)?", value)
    text = normalise(source_text)
    return bool(numbers) and all(re.search(rf"(?<![\d.,]){re.escape(n)}", text) for n in numbers)


def names_company(name: str, text: str) -> bool:
    """True if the text names the company as a whole word: 'Planes' doesn't
    count as 'Plane'. A distinctive first word also counts ('GitHub' for
    'GitHub Issues')."""
    text = normalise(text)
    full = normalise(name)
    names = {full}
    if len(full.split()[0]) >= 4:
        names.add(full.split()[0])
    return any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text) for n in names)


def _check_claim(claim: Claim, state: BranchState, today: date) -> Claim:
    competitor = state["competitor"]
    evidence = state.get("evidence", {})
    window = state.get("news_days", 90)
    source = next((evidence[e] for e in claim.evidence_ids if e in evidence), None)
    checks: list[str] = []

    def exclude(reason: str) -> Claim:
        return claim.model_copy(update={"excluded_reason": reason, "checks": checks})

    if source is None:
        return exclude("source missing")
    if source.source_type != "first_party" and not names_company(
        competitor.name, f"{source.title or ''} {source.excerpt}"
    ):
        return exclude(f"source doesn't mention {competitor.name}")
    match = quote_match(claim.quote, source.excerpt) if claim.quote else "missing"
    if match == "missing":
        # Pricing tables often extract badly (e.g. "$6" rendered as "$66"), so a
        # price whose numbers all appear in the source is kept, at low confidence.
        if claim.field == "pricing" and numbers_in_source(claim.value or "", source.excerpt):
            return claim.model_copy(update={
                "checks": ["quote not exact; price figures found in source"], "confidence": "low",
            })
        return exclude("quote not found in source")
    checks.append("quote verified" if match == "exact" else "quote closely matches source")

    if claim.field == "news":
        if claim.published_at is None:
            checks.append("undated news")
        elif claim.published_at < today - timedelta(days=window):
            return exclude(f"news older than {window} days ({claim.published_at})")
    old_price = (
        claim.field == "pricing"
        and claim.published_at is not None
        and claim.published_at < today - timedelta(days=STALE_PRICING_DAYS)
    )
    if old_price and source.source_type != "first_party":
        return exclude(f"third-party pricing from {claim.published_at} may be outdated")
    if old_price:
        # First-party pages often carry misleading dates, so keep the price but flag it.
        checks.append(f"source dated {claim.published_at}; price may be outdated")

    if "undated news" in checks or old_price:
        confidence = "low"
    elif source.source_type == "first_party" and match == "exact":
        confidence = "high"
    else:
        confidence = "medium"
    return claim.model_copy(update={"checks": checks, "confidence": confidence})


def check_quotes(state: BranchState, deps: Deps) -> dict:
    """Code checks on every found claim: the quote is in the source, the source
    is about this competitor, and news and third-party prices are recent."""
    today = date.today()
    return {
        "claims": {
            claim_id: _check_claim(claim, state, today)
            for claim_id, claim in state.get("claims", {}).items()
            if claim.status == "found" and not claim.superseded and not claim.excluded_reason
        }
    }


def missing_fields(claims: list[Claim]) -> list[str]:
    usable = {c.field for c in claims if c.status in ("found", "not_public")}
    return [field for field in REQUIRED_FIELDS if field not in usable]


def assess_evidence(state: BranchState, deps: Deps) -> dict:
    """LLM review of one competitor's claims: conflicts, wrong-company claims, gaps."""
    competitor = state["competitor"]
    claims = active_claims(state.get("claims", {}))
    labelled = label_claims(claims)
    gaps = [f"{field}: no usable claim found" for field in missing_fields(claims)]
    messages = [
        (
            "system",
            f"You review research on one competitor before it goes into a brief. Today "
            f"is {date.today()}.\n"
            "- conflicting_labels: claims that contradict each other, such as two "
            "different prices for the same plan and billing period. Different plans, "
            "billing periods, or products are not a conflict.\n"
            "- wrong_company_labels: claims about a different company or product with "
            "a similar name. Check each source's URL and wording.\n"
            "- gaps: only fields with no usable claim, or clearly incomplete ones, such "
            "as prices for only some plans. Describe each gap specifically enough to "
            "search for it.",
        ),
        (
            "human",
            f"Competitor: {competitor.name} ({competitor.domain or 'domain unknown'})\n"
            f"{describe_scope(state.get('market'), state.get('region'))}\n\n"
            f"Claims:\n{format_claims(labelled, state.get('evidence', {}))}",
        ),
    ]
    try:
        verdict = deps.llm.structured(
            role="strong", schema=EvidenceVerdict, messages=messages,
            run_id=state["run_id"], node="assess_evidence", max_output_tokens=1500,
        )
    except LLMError as exc:
        return {"gaps": gaps, "errors": [RunError(node="assess_evidence", competitor=competitor.name, message=str(exc))]}

    updates = {}
    for label in verdict.conflicting_labels:
        if label in labelled:
            claim = labelled[label]
            updates[claim.id] = claim.model_copy(update={"status": "conflicting", "confidence": "low"})
    for label in verdict.wrong_company_labels:
        if label in labelled:
            claim = labelled[label]
            updates[claim.id] = claim.model_copy(update={"excluded_reason": "about a different company"})
    gaps += [f"{gap.field}: {gap.reason}" for gap in verdict.gaps]
    return {"claims": updates, "gaps": list(dict.fromkeys(gaps))}


def make_route_after_verifier(deps: Deps):
    def route_after_verifier(state: BranchState) -> str:
        if not state.get("gaps"):
            return END
        if state.get("retry_count", 0) >= deps.settings.max_branch_retries:
            return END
        if deps.ledger.remaining(state["run_id"], "tavily") < RETRY_MIN_CREDITS:
            return END
        return "plan_queries"

    return route_after_verifier


def issue_gate(state: ResearchState, deps: Deps) -> dict:
    """Collect problems worth a reviewer's attention after all branches finish."""
    claims = state.get("claims", {})
    statuses = state.get("branch_status", {})
    issues = []
    for competitor in research_subjects(state):
        name = competitor.name
        if statuses.get(name) == "failed":
            message = next(
                (e.message for e in reversed(state.get("errors", [])) if e.competitor == name), "unknown error"
            )
            issues.append({"competitor": name, "kind": "failed", "detail": message})
            continue
        own = [c for c in active_claims(claims) if c.competitor == name]
        for field in missing_fields(own):
            issues.append({"competitor": name, "kind": "missing", "detail": f"no usable {field} information"})
        for claim in own:
            if claim.status == "conflicting":
                label = f"{claim.field} ({claim.item})" if claim.item else claim.field
                issues.append({
                    "competitor": name, "kind": "conflict", "claim_id": claim.id,
                    "detail": f"{label}: {claim.value}",
                })
    return {"open_issues": issues}


def route_after_gate(state: ResearchState) -> str:
    return "review_issues" if state.get("open_issues") else "synthesize"


def review_issues(state: ResearchState, deps: Deps) -> dict:
    """Human checkpoint, only when issues were flagged. Resume with any of
    {"rerun": [names], "exclude_claims": [claim ids]}, {} to continue, or
    {"cancel": true}."""
    answer = interrupt({"point": "review_issues", "issues": state.get("open_issues", [])})
    decision = [Decision(point="review_issues", payload=answer)]
    if answer.get("cancel"):
        return {"cancelled_reason": "Cancelled at issue review.", "human_decisions": decision}
    claims = state.get("claims", {})
    excluded = {
        claim_id: claims[claim_id].model_copy(update={"excluded_reason": REVIEWER_EXCLUDED})
        for claim_id in answer.get("exclude_claims", [])
        if claim_id in claims
    }
    return {
        "claims": excluded,
        "rerun_competitors": answer.get("rerun", []),
        "open_issues": [],
        "human_decisions": decision,
    }
