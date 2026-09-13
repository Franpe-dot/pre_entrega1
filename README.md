# Unified Async LLM Client

Cliente asíncrono unificado para OpenAI y Anthropic. Preentrega 1 — Módulo 1
(Asyncio aplicado a Ingeniería de IA).

## Qué resuelve

Una única interfaz (`AsyncLLMManager`) para hablar con OpenAI o Anthropic sin
que el resto del código sepa cuál de los dos hay detrás:

- Todas las llamadas son `async/await` (no bloquean el event loop).
- Streaming token por token con generadores asíncronos (`async for` + `yield`).
- Entradas y configuración validadas con Pydantic (`schemas.py`).
- Errores de red / API key / rate limit se capturan y devuelven como
  resultado controlado (`ModelResponse.success=False`), nunca como crash.
- Reintentos automáticos con backoff exponencial ante rate limiting o
  problemas de conexión (`retry.py`), usando `asyncio.sleep` para no
  bloquear el loop mientras se espera.

## Estructura

```
schemas.py          # ChatMessage, ModelConfig, ModelResponse (Pydantic)
base.py              # BaseLLMClient (interfaz abstracta común)
openai_client.py      # OpenAIClient (AsyncOpenAI)
anthropic_client.py   # AnthropicClient (AsyncAnthropic)
retry.py              # Backoff exponencial reutilizable
manager.py            # AsyncLLMManager: elige el proveedor según config
main.py               # Script de prueba: modo normal + streaming
```

## Setup

Requiere Python 3.12+.

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Copiá `.env.example` a `.env` y completá al menos una API key:

```bash
copy .env.example .env
```

Variables de entorno:

| Variable            | Descripción                                      |
|---------------------|---------------------------------------------------|
| `LLM_PROVIDER`      | `openai` o `anthropic`. Cuál usa `main.py`.        |
| `OPENAI_API_KEY`    | API key de OpenAI (requerida si `LLM_PROVIDER=openai`). |
| `ANTHROPIC_API_KEY` | API key de Anthropic (requerida si `LLM_PROVIDER=anthropic`). |
| `OPENAI_MODEL`      | Modelo de OpenAI a usar (default `gpt-4o-mini`).   |
| `ANTHROPIC_MODEL`   | Modelo de Anthropic a usar (default `claude-sonnet-5`). |

## Ejecutar la prueba

```bash
python main.py
```

Hace la misma pregunta ("¿Qué es la entropía?") primero con `generate()`
(respuesta completa) y después con `stream()` (impresa token por token a
medida que llega).

## Uso programático

```python
from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig

config = ModelConfig(provider="anthropic", model="claude-sonnet-5", temperature=0.5)
manager = AsyncLLMManager(config)

messages = [ChatMessage(role="user", content="Hola, ¿quién sos?")]

response = await manager.generate(messages)   # respuesta completa
async for chunk in manager.stream(messages):  # streaming
    print(chunk, end="")
```

Cambiar de proveedor es solo cambiar `provider` en `ModelConfig` (y el
`model` correspondiente) — el resto del código no cambia.

## Nota sobre la versión del SDK de Anthropic

`requirements.txt` fija `anthropic<1.0.0`. La serie 1.x del SDK eliminó el
parámetro `temperature` de `messages.create()` (lo reemplazó por un campo
`effort`), lo que rompería la validación unificada de temperatura (0–2) que
pide este ejercicio para ambos proveedores. Con `anthropic<1.0.0` ese
parámetro sigue existiendo y todo funciona como se espera.
