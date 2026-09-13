"""Implementación del cliente unificado para Anthropic."""

from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from base import BaseLLMClient
from retry import retry_async
from schemas import ChatMessage, ModelConfig, ModelResponse

RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
)


def _split_system(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
    """Anthropic pide el system prompt aparte, no dentro de la lista de mensajes."""
    system = next((m.content for m in messages if m.role == "system"), None)
    turns = [
        {"role": m.role, "content": m.content} for m in messages if m.role != "system"
    ]
    return system, turns


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        system, turns = _split_system(messages)

        async def _call():
            return await self.client.messages.create(
                model=self.config.model,
                system=system or anthropic.NOT_GIVEN,
                messages=turns,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        try:
            response = await retry_async(_call, retryable_exceptions=RETRYABLE_ERRORS)
            text = "".join(block.text for block in response.content if block.type == "text")
            return ModelResponse(
                content=text,
                model=self.config.model,
                provider="anthropic",
            )
        except anthropic.APIError as exc:
            return ModelResponse(
                content="",
                model=self.config.model,
                provider="anthropic",
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        system, turns = _split_system(messages)
        try:
            async with self.client.messages.stream(
                model=self.config.model,
                system=system or anthropic.NOT_GIVEN,
                messages=turns,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.APIError as exc:
            yield f"[error] {type(exc).__name__}: {exc}"
