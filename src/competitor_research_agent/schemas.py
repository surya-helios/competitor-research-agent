"""Data models shared by agents, state, and the UI.

Every claim links back to evidence, and every piece of evidence links back to
a URL and retrieval date, so any statement in a brief can be traced to a source.
"""

import hashlib
from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ClaimField = Literal["pricing", "feature", "positioning", "news"]
ClaimStatus = Literal["found", "not_public", "not_found", "conflicting"]
Confidence = Literal["high", "medium", "low"]
SourceType = Literal["first_party", "third_party", "news"]
BranchStatus = Literal["pending", "running", "complete", "failed"]
ReviewPoint = Literal[
    "confirm_company", "approve_competitors", "review_issues", "approve_brief"
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _short_hash(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _slug(text: str) -> str:
    return "-".join(text.lower().split())


def make_evidence_id(competitor: str, url: str, excerpt: str) -> str:
    return f"ev-{_slug(competitor)}-{_short_hash(url, excerpt)}"


def make_claim_id(competitor: str, field: ClaimField, item: str | None = None) -> str:
    """Stable ID so a rerun replaces the same claim instead of duplicating it.

    Positioning has one claim per competitor; pricing (per plan), features,
    and news have one per item, so they need an item key.
    """
    base = f"cl-{_slug(competitor)}-{field}"
    return f"{base}-{_short_hash(_slug(item))}" if item else base


class ResearchConstraints(BaseModel):
    market: str | None = None
    region: str | None = None
    news_days: int = 90
    max_competitors: int = 3


class CompanyProfile(BaseModel):
    name: str
    domain: str | None = None
    category: str | None = None
    description: str | None = None
    found: bool = Field(
        default=True,
        description="False if the search results don't clearly describe a real company or product with this name.",
    )
    is_ambiguous: bool = False
    alternatives: list[str] = Field(
        default_factory=list,
        description="Other companies the name could refer to, when ambiguous.",
    )


class Candidate(BaseModel):
    name: str
    domain: str | None = None
    rank: int = Field(ge=1)
    is_direct: bool
    overlap_score: float = Field(ge=0, le=1)
    rationale: str
    source_urls: list[str] = Field(default_factory=list)
    # True for the company being analysed, researched as the baseline.
    is_self: bool = False


class Evidence(BaseModel):
    id: str
    competitor: str
    url: str
    title: str | None = None
    excerpt: str
    source_type: SourceType
    query: str | None = None
    retrieved_at: datetime = Field(default_factory=_utcnow)
    published_at: date | None = None


class Claim(BaseModel):
    id: str
    competitor: str
    field: ClaimField
    # Plan, feature, or headline the claim is about, e.g. "Team plan".
    item: str | None = None
    value: str | None = None
    status: ClaimStatus
    evidence_ids: list[str] = Field(default_factory=list)
    quote: str | None = Field(
        default=None, description="Exact text from the evidence that supports the value."
    )
    published_at: date | None = None
    confidence: Confidence = "low"
    # Replaced by a newer extraction of the same competitor.
    superseded: bool = False
    # Verification notes, e.g. "quote verified" or "undated news".
    checks: list[str] = Field(default_factory=list)
    # Set when verification or a reviewer removes the claim from the brief.
    excluded_reason: str | None = None

    @model_validator(mode="after")
    def _found_needs_support(self) -> "Claim":
        if self.status == "found" and not (self.value and self.evidence_ids):
            raise ValueError("a 'found' claim needs a value and at least one evidence id")
        return self


# --- LLM output schemas: what reasoning nodes ask the model to return ---------


class DiscoveryPlan(BaseModel):
    queries: list[str]


class CandidateDraft(BaseModel):
    name: str
    domain: str | None = Field(description="Website domain, e.g. 'coda.io', if known.")
    is_direct: bool = Field(
        description="Same product category, same type of buyer, and a realistic substitute."
    )
    overlap_score: float = Field(ge=0, le=1, description="Product and buyer overlap.")
    rationale: str
    source_labels: list[str] = Field(description="Labels of the sources that support this, e.g. S2.")


class CandidateRanking(BaseModel):
    candidates: list[CandidateDraft]


class PlannedQuery(BaseModel):
    query: str
    purpose: ClaimField
    use_competitor_domain: bool = Field(
        description="True to search only the competitor's own website."
    )


class ResearchPlan(BaseModel):
    queries: list[PlannedQuery]


class ClaimDraft(BaseModel):
    field: ClaimField
    item: str = Field(description="Short label: plan name, feature name, or headline.")
    value: str | None = Field(
        description=(
            "A complete statement. For pricing, include currency, unit, and period, "
            "e.g. '$10 per user/month, billed annually'."
        )
    )
    status: ClaimStatus = Field(
        description=(
            "found: the evidence states the value. "
            "not_public: the evidence says it isn't published, e.g. 'contact sales'. "
            "not_found: the evidence doesn't mention it. "
            "conflicting: sources disagree."
        )
    )
    quote: str | None = Field(description="Exact text copied from the evidence.")
    evidence_id: str | None


class ClaimDrafts(BaseModel):
    claims: list[ClaimDraft]


class Gap(BaseModel):
    field: ClaimField
    reason: str = Field(description="What is missing, specific enough to search for.")


class EvidenceVerdict(BaseModel):
    gaps: list[Gap] = Field(
        description="Only fields with no usable claim, or clearly incomplete ones "
        "(e.g. some plans' prices missing). Empty if the research is good enough."
    )
    conflicting_labels: list[str] = Field(
        description="Labels of claims that contradict each other, e.g. two prices for the same plan."
    )
    wrong_company_labels: list[str] = Field(
        description="Labels of claims about a different company or product with a similar name."
    )


class RunError(BaseModel):
    node: str
    message: str
    competitor: str | None = None
    at: datetime = Field(default_factory=_utcnow)


class Decision(BaseModel):
    """A human choice at a review point. Human decisions override agent output."""

    point: ReviewPoint
    payload: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=_utcnow)


class ComparisonRow(BaseModel):
    competitor: str
    pricing: str
    key_features: list[str] = Field(default_factory=list)
    positioning: str
    recent_news: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


class ComparisonMatrix(BaseModel):
    rows: list[ComparisonRow]
    implications: list[str] = Field(default_factory=list)


class Brief(BaseModel):
    title: str
    executive_summary: str
    body_markdown: str
    gaps: list[str] = Field(
        default_factory=list, description="What could not be found or verified."
    )
    citations: dict[str, str] = Field(
        default_factory=dict,
        description="Citation label used in the text, e.g. 'C3', mapped to its claim id.",
    )
    generated_at: datetime = Field(default_factory=_utcnow)
