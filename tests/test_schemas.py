import pytest
from pydantic import ValidationError

from competitor_research_agent.schemas import Candidate, Claim, make_claim_id, make_evidence_id


def test_found_claim_requires_value_and_evidence():
    with pytest.raises(ValidationError):
        Claim(id="c", competitor="Acme", field="pricing", status="found", value="$10")
    with pytest.raises(ValidationError):
        Claim(id="c", competitor="Acme", field="pricing", status="found", evidence_ids=["e"])


def test_missing_data_claims_need_no_value():
    claim = Claim(id="c", competitor="Acme", field="pricing", status="not_public")
    assert claim.value is None and claim.confidence == "low"


def test_claim_ids_are_stable_and_item_specific():
    assert make_claim_id("Acme Corp", "pricing") == "cl-acme-corp-pricing"
    assert make_claim_id("Acme", "feature", "SSO") == make_claim_id("acme", "feature", "sso")
    assert make_claim_id("Acme", "feature", "SSO") != make_claim_id("Acme", "feature", "API")


def test_evidence_ids_depend_on_url_and_excerpt():
    first = make_evidence_id("Acme", "https://acme.com/pricing", "Pro: $10")
    assert first == make_evidence_id("Acme", "https://acme.com/pricing", "Pro: $10")
    assert first != make_evidence_id("Acme", "https://acme.com/pricing", "Pro: $12")


def test_candidate_score_is_bounded():
    with pytest.raises(ValidationError):
        Candidate(name="Acme", rank=1, is_direct=True, overlap_score=1.5, rationale="x")
