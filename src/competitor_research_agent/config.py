"""Run settings, loaded from environment variables (and .env if present)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelConfig:
    """One Nebius model and its price, used to estimate and record spend."""

    model_id: str | None
    usd_per_m_input: float = 0.0
    usd_per_m_output: float = 0.0


@dataclass(frozen=True)
class Settings:
    nebius_api_key: str | None = None
    tavily_api_key: str | None = None
    nebius_base_url: str = NEBIUS_BASE_URL

    # Model roles: fast for planning/extraction, strong for judgement/writing,
    # backup when a primary model fails.
    fast_model: ModelConfig = field(default_factory=lambda: ModelConfig(None))
    strong_model: ModelConfig = field(default_factory=lambda: ModelConfig(None))
    backup_model: ModelConfig = field(default_factory=lambda: ModelConfig(None))

    # Per-run limits
    # Covers the company plus 3 competitors, with retries.
    tavily_credit_limit: float = 90
    nebius_usd_limit: float = 0.50
    max_branch_retries: int = 2
    max_concurrency: int = 3
    max_competitors: int = 3
    news_days: int = 90

    # Cache expiry, in hours
    cache_ttl_news_hours: int = 24
    cache_ttl_pages_hours: int = 24 * 7
    cache_ttl_discovery_hours: int = 24 * 7

    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def app_db(self) -> Path:
        return self.data_dir / "app.sqlite"


def _model_from_env(role: str) -> ModelConfig:
    prefix = f"NEBIUS_MODEL_{role.upper()}"
    return ModelConfig(
        model_id=os.getenv(prefix) or None,
        usd_per_m_input=float(os.getenv(f"{prefix}_USD_PER_M_INPUT", "0")),
        usd_per_m_output=float(os.getenv(f"{prefix}_USD_PER_M_OUTPUT", "0")),
    )


def load_settings(env_file: Path | None = None) -> Settings:
    """Build settings from the environment. Missing keys are allowed here;
    the startup check reports them before a run spends anything."""
    load_dotenv(env_file or PROJECT_ROOT / ".env")
    defaults = Settings()
    return Settings(
        nebius_api_key=os.getenv("NEBIUS_API_KEY") or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        nebius_base_url=os.getenv("NEBIUS_BASE_URL", NEBIUS_BASE_URL),
        fast_model=_model_from_env("fast"),
        strong_model=_model_from_env("strong"),
        backup_model=_model_from_env("backup"),
        tavily_credit_limit=float(
            os.getenv("TAVILY_CREDIT_LIMIT", defaults.tavily_credit_limit)
        ),
        nebius_usd_limit=float(os.getenv("NEBIUS_USD_LIMIT", defaults.nebius_usd_limit)),
        data_dir=Path(os.getenv("DATA_DIR", defaults.data_dir)),
    )
