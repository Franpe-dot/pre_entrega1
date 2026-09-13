"""Implementación del cliente unificado para OpenAI."""

from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from base import BaseLLMClient
from retry import retry_async
from schemas import ChatMessage, ModelConfig, ModelResponse

RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        payload = [m.model_dump() for m in messages]

        async def _call():
            return await self.client.chat.completions.create(
                model=self.config.model,
                messages=payload,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        try:
            response = await retry_async(_call, retryable_exceptions=RETRYABLE_ERRORS)
            return ModelResponse(
                content=response.choices[0].message.content or "",
                model=self.config.model,
                provider="openai",
            )
        except openai.APIError as exc:
            return ModelResponse(
                content="",
                model=self.config.model,
                provider="openai",
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        payload = [m.model_dump() for m in messages]
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=payload,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except openai.APIError as exc:
            yield f"[error] {type(exc).__name__}: {exc}"
