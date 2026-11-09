"""Helpers shared by agents: dates, source types, and citation labels."""

import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from ..schemas import Candidate, Claim, CompanyProfile, Evidence

# Matches [C3] and grouped citations like [C1, C6].
_CITATION = re.compile(r"\[(C\d+(?:\s*,\s*C\d+)*)\]")


def baseline_candidate(company: CompanyProfile) -> Candidate:
    """The company being analysed, researched like a competitor so the brief
    can compare against it."""
    return Candidate(
        name=company.name, domain=company.domain, rank=1, is_direct=False,
        overlap_score=1.0, rationale="The company being analysed (baseline).", is_self=True,
    )


def research_subjects(state: dict) -> list[Candidate]:
    """Everyone researched in a run: the company first, then approved competitors."""
    return [baseline_candidate(state["company"]), *state.get("approved_competitors", [])]


def describe_scope(market: str | None, region: str | None) -> str:
    """Prompt lines stating the market and region the user asked about."""
    return (
        f"Market: {market or 'not specified'}\n"
        f"Region: {region or 'global (not specified)'}"
    )


def parse_published(value: str | None) -> date | None:
    """Tavily returns dates like 'Fri, 18 Aug 2023 00:00:00 GMT' or ISO strings."""
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def is_first_party(url: str, domain: str | None) -> bool:
    if not domain:
        return False
    host = urlparse(url).netloc.lower().removeprefix("www.")
    domain = domain.lower().removeprefix("www.")
    return host == domain or host.endswith("." + domain)


def label_claims(claims: list[Claim]) -> dict[str, Claim]:
    """Short labels (C1, C2, ...) for prompts. Models copy these more reliably
    than long claim ids."""
    ordered = sorted(claims, key=lambda c: (c.competitor, c.field, c.id))
    return {f"C{i}": claim for i, claim in enumerate(ordered, start=1)}


def format_claims(labelled: dict[str, Claim], evidence: dict[str, Evidence]) -> str:
    lines = []
    for label, claim in labelled.items():
        sources = [evidence[eid] for eid in claim.evidence_ids if eid in evidence]
        source = (
            f"{sources[0].source_type}, {sources[0].url}"
            + (f", published {sources[0].published_at}" if sources[0].published_at else "")
            if sources
            else "no source"
        )
        field = f"{claim.field} ({claim.item})" if claim.item else claim.field
        lines.append(
            f"[{label}] {claim.competitor} | {field} | {claim.value or '-'} "
            f"| status: {claim.status} | {source}"
        )
    return "\n".join(lines)


def cited_labels(*texts: str) -> list[str]:
    seen = dict.fromkeys(
        label.strip()
        for text in texts
        for group in _CITATION.findall(text)
        for label in group.split(",")
    )
    return list(seen)
