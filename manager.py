"""Punto de entrada único: elige el proveedor real según la configuración."""

import os
from collections.abc import AsyncIterator

from anthropic_client import AnthropicClient
from base import BaseLLMClient
from openai_client import OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class AsyncLLMManager:
    """Fachada asíncrona intercambiable entre proveedores de LLM."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._client: BaseLLMClient = self._build_client(config)

    @staticmethod
    def _build_client(config: ModelConfig) -> BaseLLMClient:
        if config.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Falta OPENAI_API_KEY en el entorno (.env)")
            return OpenAIClient(config, api_key=api_key)

        if config.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("Falta ANTHROPIC_API_KEY en el entorno (.env)")
            return AnthropicClient(config, api_key=api_key)

        raise ValueError(f"Proveedor no soportado: {config.provider}")

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        return await self._client.generate(messages)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async for chunk in self._client.stream(messages):
            yield chunk
