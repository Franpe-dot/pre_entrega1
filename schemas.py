"""Modelos Pydantic para el cliente unificado de LLMs."""

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]
Provider = Literal["openai", "anthropic"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class ModelConfig(BaseModel):
    provider: Provider
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)


class ModelResponse(BaseModel):
    content: str
    model: str
    provider: Provider
    success: bool = True
    error: str | None = None
