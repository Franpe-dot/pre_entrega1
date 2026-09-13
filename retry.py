"""Reintentos con backoff exponencial para llamadas a APIs externas.

Usa asyncio.sleep (nunca time.sleep) para no bloquear el event loop
mientras se espera entre reintentos.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    retryable_exceptions: tuple[type[Exception], ...],
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> T:
    last_exc: Exception
    for attempt in range(max_attempts):
        try:
            return await func()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            await asyncio.sleep(base_delay * (2**attempt))
    raise last_exc
