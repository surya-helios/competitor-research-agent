"""Validate user input before a run spends anything.

These rules only catch obvious junk. Whether a plausible name is a real
company is decided later from search results (CompanyProfile.found).
"""

import re
from urllib.parse import urlparse

_DOMAIN = re.compile(r"^(?=.{4,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_NAME_CHARS = re.compile(r"[\w\s.&'+,/()-]", re.UNICODE)


class InvalidInput(ValueError):
    """Input rejected before any search or model call."""


def _clean_text(raw: str | None, what: str) -> str:
    text = " ".join((raw or "").split())
    if not 2 <= len(text) <= 100:
        raise InvalidInput(f"Enter a {what} between 2 and 100 characters.")
    if not any(ch.isalpha() for ch in text):
        raise InvalidInput(f"A {what} needs at least one letter.")
    usual = sum(1 for ch in text if _NAME_CHARS.match(ch))
    if usual / len(text) < 0.8:
        raise InvalidInput(f"That doesn't look like a {what}: it's mostly symbols.")
    return text


def clean_company_name(raw: str) -> str:
    return _clean_text(raw, "company name")


def clean_market(raw: str | None) -> str | None:
    """Optional market or industry, e.g. 'issue tracking software'."""
    return _clean_text(raw, "market") if (raw or "").strip() else None


def clean_region(raw: str | None) -> str | None:
    """Optional region, e.g. 'India', 'EU', or 'North America'."""
    return _clean_text(raw, "region") if (raw or "").strip() else None


def clean_domain(raw: str | None) -> str | None:
    """Accept 'plane.so', 'www.plane.so', or 'https://plane.so/pricing'; return 'plane.so'."""
    text = (raw or "").strip().lower()
    if not text:
        return None
    host = urlparse(text if "://" in text else f"https://{text}").hostname or ""
    host = host.removeprefix("www.")
    if not _DOMAIN.match(host):
        raise InvalidInput(f"'{raw}' isn't a valid website. Use a domain like example.com.")
    return host
