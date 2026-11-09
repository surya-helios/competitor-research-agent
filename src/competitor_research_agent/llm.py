"""Nebius LLM access for reasoning nodes.

`LLMClient.structured()` is the only way nodes call a model. It returns a
validated Pydantic object or raises LLMError, trying in order:

1. JSON mode, with the schema in the prompt, then Pydantic validation.
   (Plain prompting if the model doesn't support JSON mode.)
2. One repair attempt that sends the validation error back to the model.
3. The same steps on the backup model.

Native JSON-schema output is deliberately not used: on Nebius, Qwen models
returned the same enum value for every item in that mode, while JSON mode,
function calling, and plain prompting all labelled them correctly.

The OpenAI SDK already retries rate limits, timeouts, and 5xx errors. Every
call reserves its worst-case cost in the budget ledger and settles actual
token usage afterwards.
"""

import json
import logging
import re
from typing import Any, Callable, Literal

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from .budget import BudgetExceeded, BudgetLedger, llm_cost_usd
from .config import ModelConfig, Settings

Role = Literal["fast", "strong"]
Messages = list[tuple[str, str]]
ChatFactory = Callable[[ModelConfig, int], BaseChatModel]

# Errors no fallback can fix: stop the run and tell the user.
FATAL_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError, BudgetExceeded)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

log = logging.getLogger(__name__)


class LLMError(Exception):
    """No model produced valid output."""


class LLMOutputError(Exception):
    """A model responded, but its output didn't match the schema."""


def parse_json_object(text: str) -> Any:
    """Pull a JSON object out of model text that may include reasoning blocks,
    code fences, or a sentence before or after the JSON."""
    text = _THINK_BLOCK.sub("", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMOutputError("no JSON object in response")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"invalid JSON: {exc}") from exc


def _estimate_tokens(messages: Messages) -> int:
    # About 4 characters per token for English; 3 keeps the reservation on the safe side.
    return sum(len(content) for _, content in messages) // 3 + 1


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        ledger: BudgetLedger,
        chat_factory: ChatFactory | None = None,
    ):
        self._settings = settings
        self._ledger = ledger
        self._chat_factory = chat_factory or self._default_chat

    def _default_chat(self, model: ModelConfig, max_output_tokens: int) -> BaseChatModel:
        return ChatOpenAI(
            model=model.model_id,
            api_key=self._settings.nebius_api_key,
            base_url=self._settings.nebius_base_url,
            temperature=0,
            max_tokens=max_output_tokens,
            max_retries=2,
            timeout=90,
        )

    def models_for(self, role: Role) -> list[ModelConfig]:
        primary = self._settings.fast_model if role == "fast" else self._settings.strong_model
        backup = self._settings.backup_model
        models = [m for m in (primary, backup) if m.model_id]
        # Don't try the same model twice.
        return list({m.model_id: m for m in models}.values())

    def structured[T: BaseModel](
        self,
        *,
        role: Role,
        schema: type[T],
        messages: Messages,
        run_id: str,
        node: str,
        max_output_tokens: int = 2000,
    ) -> T:
        models = self.models_for(role)
        if not models:
            raise LLMError(f"no model configured for role '{role}'")
        failures = []
        for model in models:
            try:
                return self._structured_with_model(
                    model, schema, messages, run_id, node, max_output_tokens
                )
            except FATAL_ERRORS:
                raise
            except Exception as exc:
                failures.append(f"{model.model_id}: {type(exc).__name__}: {exc}")
                log.warning("%s failed on %s, trying next model: %s", model.model_id, node, failures[-1])
        raise LLMError("all models failed: " + " | ".join(failures))

    def _structured_with_model[T: BaseModel](
        self,
        model: ModelConfig,
        schema: type[T],
        messages: Messages,
        run_id: str,
        node: str,
        max_output_tokens: int,
    ) -> T:
        chat = self._chat_factory(model, max_output_tokens)
        json_chat = chat.bind(response_format={"type": "json_object"})
        prompted = messages + [
            (
                "human",
                "Respond with only a JSON object that matches this JSON schema:\n"
                + json.dumps(schema.model_json_schema()),
            )
        ]

        def ask(runnable, prompt: Messages) -> AIMessage:
            return self._metered(
                model, run_id, node, prompt, max_output_tokens, lambda: runnable.invoke(prompt)
            )

        try:
            reply = ask(json_chat, prompted)
        except openai.BadRequestError:
            json_chat = chat  # Model doesn't support JSON mode; rely on the prompt.
            reply = ask(json_chat, prompted)

        try:
            return schema.model_validate(parse_json_object(reply.text))
        except (LLMOutputError, ValidationError) as exc:
            repair = prompted + [
                ("ai", reply.text),
                ("human", f"That output was invalid: {exc}\nReturn only the corrected JSON object."),
            ]
            repaired = ask(json_chat, repair)
            try:
                return schema.model_validate(parse_json_object(repaired.text))
            except ValidationError as final:
                raise LLMOutputError(str(final)) from final

    def _metered(
        self,
        model: ModelConfig,
        run_id: str,
        node: str,
        messages: Messages,
        max_output_tokens: int,
        call: Callable[[], AIMessage],
    ) -> AIMessage:
        worst_case = llm_cost_usd(
            _estimate_tokens(messages),
            max_output_tokens,
            model.usd_per_m_input,
            model.usd_per_m_output,
        )
        entry = self._ledger.reserve(run_id, "nebius", worst_case, node=node, model=model.model_id)
        try:
            result = call()
        except BaseException:
            # Failed requests aren't billed; keep the row for the call count.
            self._ledger.settle(entry, 0)
            raise
        usage = result.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        self._ledger.settle(
            entry,
            llm_cost_usd(input_tokens, output_tokens, model.usd_per_m_input, model.usd_per_m_output),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return result
