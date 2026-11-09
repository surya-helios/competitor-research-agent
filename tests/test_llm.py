import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from competitor_research_agent.budget import BudgetLedger
from competitor_research_agent.config import ModelConfig, Settings
from competitor_research_agent.llm import LLMClient, LLMError, LLMOutputError, parse_json_object


class Verdict(BaseModel):
    sufficient: bool
    gaps: list[str]


VALID = '{"sufficient": true, "gaps": []}'


def _api_error(cls, status):
    request = httpx.Request("POST", "https://example.test")
    return cls("error", response=httpx.Response(status, request=request), body=None)


def _reply(text, input_tokens=100, output_tokens=20):
    return AIMessage(
        content=text,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


class FakeChat:
    """Stands in for ChatOpenAI. Each invoke() returns or raises the next item
    in `replies`. If `json_mode_error` is set, JSON-mode calls raise it."""

    def __init__(self, replies=(), json_mode_error=None):
        self.replies = list(replies)
        self.json_mode_error = json_mode_error
        self.calls = []

    def bind(self, **kwargs):
        chat = self

        class JsonMode:
            def invoke(self, messages):
                if chat.json_mode_error:
                    chat.calls.append(("json", messages))
                    raise chat.json_mode_error
                return chat._next("json", messages)

        return JsonMode()

    def invoke(self, messages):
        return self._next("plain", messages)

    def _next(self, mode, messages):
        self.calls.append((mode, messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def setup(tmp_path):
    settings = Settings(
        fast_model=ModelConfig("org/fast", 1.0, 2.0),
        backup_model=ModelConfig("org/backup", 1.0, 2.0),
    )
    ledger = BudgetLedger(tmp_path / "app.sqlite", {"tavily": 10, "nebius": 1})

    def make(chats: dict[str, FakeChat]) -> LLMClient:
        return LLMClient(settings, ledger, chat_factory=lambda model, _: chats[model.model_id])

    return make, ledger


def _call(llm):
    return llm.structured(
        role="fast", schema=Verdict, messages=[("human", "judge")], run_id="run-1", node="assess"
    )


def test_json_mode_with_schema_in_prompt(setup):
    make, ledger = setup
    chat = FakeChat([_reply(VALID)])
    assert _call(make({"org/fast": chat})).sufficient is True
    mode, prompt = chat.calls[0]
    assert mode == "json" and '"sufficient"' in prompt[-1][1]
    # 100 input tokens at $1/M plus 20 output tokens at $2/M
    assert ledger.spent("run-1", "nebius") == pytest.approx(0.00014)


def test_plain_prompt_when_json_mode_unsupported(setup):
    make, _ = setup
    chat = FakeChat(
        [_reply('<think>hmm</think>\n```json\n{"sufficient": false, "gaps": ["pricing"]}\n```')],
        json_mode_error=_api_error(openai.BadRequestError, 400),
    )
    assert _call(make({"org/fast": chat})).gaps == ["pricing"]
    assert [mode for mode, _ in chat.calls] == ["json", "plain"]


def test_invalid_json_gets_one_repair_attempt(setup):
    make, _ = setup
    chat = FakeChat([_reply('{"sufficient": "maybe"}'), _reply(VALID)])
    assert _call(make({"org/fast": chat})).sufficient is True
    assert "invalid" in chat.calls[1][1][-1][1]


def test_backup_model_used_when_primary_fails(setup):
    make, _ = setup
    chats = {
        "org/fast": FakeChat([_api_error(openai.InternalServerError, 500)]),
        "org/backup": FakeChat([_reply(VALID)]),
    }
    assert _call(make(chats)).sufficient is True


def test_all_models_failing_raises_llm_error(setup):
    make, _ = setup
    chats = {name: FakeChat([_reply("no json"), _reply("still none")]) for name in ("org/fast", "org/backup")}
    with pytest.raises(LLMError, match="all models failed"):
        _call(make(chats))


def test_auth_error_stops_without_fallback(setup):
    make, _ = setup
    chats = {
        "org/fast": FakeChat([_api_error(openai.AuthenticationError, 401)]),
        "org/backup": FakeChat([_reply(VALID)]),
    }
    with pytest.raises(openai.AuthenticationError):
        _call(make(chats))
    assert chats["org/backup"].calls == []


def test_failed_call_is_not_charged(setup):
    make, ledger = setup
    chat = FakeChat([_api_error(openai.AuthenticationError, 401)])
    with pytest.raises(openai.AuthenticationError):
        _call(make({"org/fast": chat, "org/backup": chat}))
    assert ledger.spent("run-1", "nebius") == 0


def test_parse_json_object_rejects_text_without_json():
    with pytest.raises(LLMOutputError):
        parse_json_object("I could not find anything.")
