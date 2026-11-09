"""Checks that run before a research run spends anything.

Run directly to check your setup:  uv run python -m competitor_research_agent.startup
"""

from dataclasses import dataclass, replace

import openai
import requests
from pydantic import BaseModel

from .budget import BudgetLedger
from .config import Settings, load_settings
from .llm import LLMClient

TAVILY_USAGE_URL = "https://api.tavily.com/usage"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


class _Ping(BaseModel):
    status: str


def check_tavily(settings: Settings) -> CheckResult:
    if not settings.tavily_api_key:
        return CheckResult("Tavily key", False, "TAVILY_API_KEY is not set")
    try:
        response = requests.get(
            TAVILY_USAGE_URL,
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            timeout=15,
        )
    except requests.RequestException as exc:
        return CheckResult("Tavily key", False, f"request failed: {exc}")
    if response.status_code != 200:
        return CheckResult("Tavily key", False, f"HTTP {response.status_code}")
    account = response.json().get("account", {})
    return CheckResult(
        "Tavily key",
        True,
        f"plan {account.get('current_plan')}, "
        f"plan usage {account.get('plan_usage')}/{account.get('plan_limit')}, "
        f"pay-as-you-go usage {account.get('paygo_usage')}/{account.get('paygo_limit')}",
    )


def list_nebius_models(settings: Settings) -> list[str]:
    client = openai.OpenAI(api_key=settings.nebius_api_key, base_url=settings.nebius_base_url)
    return sorted(model.id for model in client.models.list())


def check_nebius(settings: Settings) -> list[CheckResult]:
    if not settings.nebius_api_key:
        return [CheckResult("Nebius key", False, "NEBIUS_API_KEY is not set")]
    try:
        available = set(list_nebius_models(settings))
    except openai.APIError as exc:
        return [CheckResult("Nebius key", False, f"{type(exc).__name__}: {exc}")]
    results = [CheckResult("Nebius key", True, f"{len(available)} models available")]

    for role in ("fast", "strong", "backup"):
        model = getattr(settings, f"{role}_model")
        if not model.model_id:
            results.append(CheckResult(f"{role} model", False, "not configured yet"))
        elif model.model_id not in available:
            results.append(
                CheckResult(f"{role} model", False, f"{model.model_id} is not available")
            )
        elif not (model.usd_per_m_input and model.usd_per_m_output):
            results.append(
                CheckResult(f"{role} model", False, "prices not set, so spend can't be limited")
            )
        else:
            results.append(CheckResult(f"{role} model", True, model.model_id))
    return results


def check_structured_output(settings: Settings, ledger: BudgetLedger) -> list[CheckResult]:
    """One tiny structured call per configured model; costs a fraction of a cent."""
    results = []
    for role in ("fast", "strong", "backup"):
        model = getattr(settings, f"{role}_model")
        if not model.model_id:
            continue
        # Test this model alone, with no fallback to hide a failure.
        single = replace(settings, fast_model=model, backup_model=model)
        try:
            LLMClient(single, ledger).structured(
                role="fast",
                schema=_Ping,
                messages=[("human", 'Reply with status "ok".')],
                run_id="startup-check",
                node="startup_check",
                max_output_tokens=200,
            )
            results.append(CheckResult(f"{role} structured output", True, model.model_id))
        except Exception as exc:
            results.append(
                CheckResult(f"{role} structured output", False, f"{type(exc).__name__}: {exc}")
            )
    return results


def run_checks(settings: Settings, include_structured: bool = True) -> list[CheckResult]:
    results = [check_tavily(settings), *check_nebius(settings)]
    if include_structured and all(r.ok for r in results if r.name == "Nebius key"):
        ledger = BudgetLedger(
            settings.app_db,
            {"tavily": settings.tavily_credit_limit, "nebius": settings.nebius_usd_limit},
        )
        results += check_structured_output(settings, ledger)
    return results


if __name__ == "__main__":
    for result in run_checks(load_settings()):
        print(f"[{'ok' if result.ok else 'FAIL'}] {result.name}: {result.detail}")
