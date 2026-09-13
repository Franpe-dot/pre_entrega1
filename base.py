"""Interfaz común (abstracta) que deben implementar todos los proveedores."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    """Contrato asíncrono que unifica cualquier proveedor de LLM."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Devuelve la respuesta completa del modelo (no bloqueante)."""
        raise NotImplementedError

    @abstractmethod
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Generador asíncrono que va devolviendo fragmentos de texto."""
        raise NotImplementedError
