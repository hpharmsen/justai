# Architecture

JustAI biedt één `Model`-klasse als toegangspoort tot LLM's van verschillende providers. De model-naam bepaalt welke provider gebruikt wordt.

## Componenten

### `justai/model/`
- **`model.py`** — Publieke `Model`-klasse. Delegeert alle calls naar een provider-specifieke implementatie die door de factory wordt aangemaakt.
- **`message.py`** — `Message` en `ToolUseMessage` dataclasses; wire-format neutraal.

### `justai/models/`
Provider-implementaties. Elk erft van `BaseModel` (of van een tussenklasse zoals `OpenAICompletionsModel`) en implementeert `prompt`, `chat`, `prompt_async`, `chat_async` en `token_count`.

- **`basemodel.py`** — Abstracte basisklasse + gedeelde exceptions (`ConnectionException`, `RatelimitException`, `RefusalException`, ...). Definieert het `effort`-mechanisme (universele levels `low/medium/high/xhigh/max` met per-provider downmap-warnings).
- **`openai_responses.py`** — OpenAI GPT-5, o1, o3 via de Responses API.
- **`openai_completions.py`** — OpenAI Chat Completions wire-format. Basis voor providers die OpenAI-compatible zijn.
- **`anthropic_models.py`** — Claude (met prompt caching + native effort).
- **`google_models.py`** — Gemini.
- **`xai_models.py`** — Grok.
- **`deepseek_models.py`** — DeepSeek (erft van `OpenAICompletionsModel`).
- **`perplexity_models.py`** — Sonar.
- **`reve_models.py`** — Reve (image generation).
- **`openrouter_models.py`** — OpenRouter aggregator.
- **`kimi_models.py`** — Kimi / Moonshot (erft van `OpenAICompletionsModel`).
- **`gguf_models.py`** — Lokale GGUF-modellen via `llama-cpp-python`.
- **`modelfactory.py`** — `ModelFactory.create()` matcht op prefix en retourneert de juiste subclass.

### `justai/tools/`
- **`cache.py`** — Lokale response-cache (`cached=True`), singleton `CacheDB`.
- **`images.py`** — PIL/URL/raw → base64 helper voor multimodale calls.
- **`display.py`** — Kleuroutput voor errors en debug.
- **`prompts.py`** — Prompt-loader helpers.

## Model-naam → provider routing

Volgorde uit `ModelFactory.create`:

| Prefix / suffix | Provider klasse |
|---|---|
| `openrouter/*` | `OpenRouterModel` |
| `gpt*`, `o1*`, `o3*` | `OpenAIResponsesModel` |
| `*.gguf` | `GgufModel` |
| `claude*` | `AnthropicModel` |
| `gemini*` | `GoogleModel` |
| `grok*` | `XAIModel` |
| `deepseek*` | `DeepSeekModel` |
| `sonar*` | `PerplexityModel` |
| `reve*` | `ReveModel` |
| `kimi*`, `moonshot*` | `KimiModel` |

## Uitbreidbaarheid

Nieuwe provider toevoegen:
1. Maak `justai/models/<provider>_models.py` met een subclass van `BaseModel` (of van `OpenAICompletionsModel` als de wire-format OpenAI-compatible is).
2. Voeg prefix-match toe in `ModelFactory.create`.
3. Documenteer de prefix in `README.md` (features + prefix-tabel), `CLAUDE.md` (Model Factory-lijst) en optioneel keywords in `pyproject.toml`.
4. Schrijf een `tests/test_<provider>.py` met factory- en init-tests (kopieer de `isolate_cache` fixture uit `test_effort.py`).

## Message flow

```
Model.prompt("hi")
  → self.model.prompt(...)                        # BaseModel subclass
  → self.completion(...)                          # bouwt messages, kiest streaming vs parse
  → client.chat.completions.create(...)           # OpenAI-compatible providers
  → tool-loop indien tool_calls aanwezig
  → response.choices[0].message.content
```

Async variant (`prompt_async`) yieldt `(content, reasoning)`-tuples per delta-chunk.

## Cross-cutting features

- **Prompt caching** — Bij Anthropic standaard aan: `apply_cache_control()` in
  `anthropic_models.py` zet per request twee breakpoints, na de systeemprompt (dat
  dekt door de rendervolgorde meteen de tool-definities) en na de laatste beurt,
  vanaf de tweede message. Copy-on-write, want de Agent hergebruikt zijn
  messages-lijst tussen iteraties. Te sturen met `cache_ttl` en `prompt_cache`.
  OpenAI en Gemini cachen server-side vanzelf. De cachetellers
  (`cache_read_input_tokens`, `cache_creation_input_tokens`) staan op `BaseModel`
  en worden gezet door `record_usage()`, dus elke provider heeft ze.
- **Effort** — Universeel API (`effort='high'`), met warning-based downmap per provider.
- **Tool use** — Uniforme `add_tool()` API; provider vertaalt naar eigen function-calling spec.
- **JSON / structured output** — `return_json=True` of `response_format=PydanticModel`. Structured output is voorlopig alleen native op OpenAI.
- **Multimodaal** — PIL Image, URL of raw bytes → automatische base64-encoding.
