from competitor_research_agent.config import NEBIUS_BASE_URL, load_settings

ENV_VARS = (
    "NEBIUS_API_KEY",
    "TAVILY_API_KEY",
    "NEBIUS_BASE_URL",
    "NEBIUS_MODEL_FAST",
    "NEBIUS_MODEL_FAST_USD_PER_M_INPUT",
    "TAVILY_CREDIT_LIMIT",
    "NEBIUS_USD_LIMIT",
    "DATA_DIR",
)


def _clear_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_without_env(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    settings = load_settings(env_file=tmp_path / "missing.env")
    assert settings.nebius_api_key is None
    assert settings.nebius_base_url == NEBIUS_BASE_URL
    assert settings.tavily_credit_limit == 90
    assert settings.nebius_usd_limit == 0.5
    assert settings.fast_model.model_id is None


def test_reads_env_file(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NEBIUS_API_KEY=nb-key\n"
        "NEBIUS_MODEL_FAST=org/fast-model\n"
        "NEBIUS_MODEL_FAST_USD_PER_M_INPUT=0.13\n"
        "TAVILY_CREDIT_LIMIT=40\n"
        f"DATA_DIR={tmp_path / 'data'}\n"
    )
    settings = load_settings(env_file=env_file)
    assert settings.nebius_api_key == "nb-key"
    assert settings.fast_model.model_id == "org/fast-model"
    assert settings.fast_model.usd_per_m_input == 0.13
    assert settings.tavily_credit_limit == 40
    assert settings.app_db == tmp_path / "data" / "app.sqlite"
