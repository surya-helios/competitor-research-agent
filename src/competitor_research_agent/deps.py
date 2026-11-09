"""Shared clients that graph nodes need, built once per process."""

from dataclasses import dataclass

from .budget import BudgetLedger
from .config import Settings, load_settings
from .llm import LLMClient
from .memory import CompetitorMemory
from .tools.cache import SearchCache
from .tools.tavily_client import TavilyTool


@dataclass(frozen=True)
class Deps:
    settings: Settings
    ledger: BudgetLedger
    llm: LLMClient
    tavily: TavilyTool
    memory: CompetitorMemory


def build_deps(settings: Settings | None = None) -> Deps:
    settings = settings or load_settings()
    ledger = BudgetLedger(
        settings.app_db,
        {"tavily": settings.tavily_credit_limit, "nebius": settings.nebius_usd_limit},
    )
    return Deps(
        settings=settings,
        ledger=ledger,
        llm=LLMClient(settings, ledger),
        tavily=TavilyTool(settings, ledger, SearchCache(settings.app_db)),
        memory=CompetitorMemory(settings.app_db),
    )
