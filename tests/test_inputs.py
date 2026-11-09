import pytest

from competitor_research_agent.inputs import (
    InvalidInput,
    clean_company_name,
    clean_domain,
    clean_market,
    clean_region,
)


@pytest.mark.parametrize("value", ["issue tracking software", "B2B SaaS (HR/payroll)", "EV charging"])
def test_markets_are_accepted(value):
    assert clean_market(value) == value


@pytest.mark.parametrize("value", ["India", "North America", "EU", "Asia-Pacific", "São Paulo, Brazil"])
def test_regions_are_accepted(value):
    assert clean_region(value) == value


def test_market_and_region_are_optional():
    assert clean_market(None) is None and clean_market("   ") is None
    assert clean_region("") is None


@pytest.mark.parametrize("junk", ["1234", "#@!$%^", "x"])
def test_junk_market_and_region_are_rejected(junk):
    with pytest.raises(InvalidInput):
        clean_market(junk)
    with pytest.raises(InvalidInput):
        clean_region(junk)


@pytest.mark.parametrize("name", ["Notion", "AT&T", "3M", "Monday.com", "L'Oréal", "Procter & Gamble", "  Coda  "])
def test_real_names_are_accepted(name):
    assert clean_company_name(name) == " ".join(name.split())


@pytest.mark.parametrize("junk", ["", "   ", "a", "123456", "!!!???", "a#@!$%^&*", "x" * 101])
def test_junk_is_rejected(junk):
    with pytest.raises(InvalidInput):
        clean_company_name(junk)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plane.so", "plane.so"),
        ("https://www.Plane.so/pricing", "plane.so"),
        ("WWW.coda.io", "coda.io"),
        ("app.linear.app", "app.linear.app"),
        ("", None),
        (None, None),
    ],
)
def test_websites_are_normalised(raw, expected):
    assert clean_domain(raw) == expected


@pytest.mark.parametrize("bad", ["not a website", "localhost", "http://", "plane", "exa mple.com"])
def test_invalid_websites_are_rejected(bad):
    with pytest.raises(InvalidInput):
        clean_domain(bad)
