"""Script de prueba: una misma pregunta en modo normal y en streaming,
contra el proveedor que se indique en LLM_PROVIDER (.env)."""

import asyncio
import os

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig

QUESTION = "¿Qué es la entropía? Respondé en dos oraciones."


async def run_normal(manager: AsyncLLMManager, messages: list[ChatMessage]) -> None:
    print("\n--- Modo normal (generate) ---")
    response = await manager.generate(messages)
    if response.success:
        print(f"[{response.provider}/{response.model}] {response.content}")
    else:
        print(f"[{response.provider}] ERROR: {response.error}")


async def run_streaming(manager: AsyncLLMManager, messages: list[ChatMessage]) -> None:
    print("\n--- Modo streaming (stream) ---")
    async for chunk in manager.stream(messages):
        print(chunk, end="", flush=True)
    print()


async def main() -> None:
    load_dotenv()

    provider = os.environ.get("LLM_PROVIDER", "openai")
    model = (
        os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if provider == "openai"
        else os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    )

    config = ModelConfig(provider=provider, model=model, temperature=0.7, max_tokens=200)
    manager = AsyncLLMManager(config)

    messages = [
        ChatMessage(role="system", content="Sos un asistente conciso y claro."),
        ChatMessage(role="user", content=QUESTION),
    ]

    await run_normal(manager, messages)
    await run_streaming(manager, messages)


if __name__ == "__main__":
    asyncio.run(main())
