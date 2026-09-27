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
- **`minimax_models.py`** — MiniMax.
- **`systemone.py`** — System One-classificatie: het wire-formaat als losse functies, plus `SystemOneMixin` (de HTTP-helft) en `SystemOneModel` (TypeSafe en zelf-gehost). Geen tekstgeneratie; `prompt` en vrienden gooien `NotImplementedError`.
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
| `minimax*` (case-insensitief) | `MiniMaxModel` |
| `systemone/*`, `jev*` | `SystemOneModel` |

## Uitbreidbaarheid

Nieuwe provider toevoegen:
1. Maak `justai/models/<provider>_models.py` met een subclass van `BaseModel` (of van `OpenAICompletionsModel` als de wire-format OpenAI-compatible is).
2. Voeg prefix-match toe in `ModelFactory.create`.
3. Documenteer de prefix in `README.md` (features + prefix-tabel), `CLAUDE.md` en `AGENTS.md` (beide hebben dezelfde Model Factory-lijst) en optioneel keywords in `pyproject.toml`.
4. Schrijf een `tests/test_<provider>.py` met factory- en init-tests (kopieer de `isolate_cache_and_warnings` fixture uit `test_effort.py`).

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
- **JSON / structured output**: `return_json=True` of `response_format=PydanticModel`. Native
  structured output bij Anthropic en OpenAI. Bij een Pydantic-klasse geven de providers de
  rauwe JSON-tekst terug en valideert `Model` centraal via `_to_pydantic`; de SDK-`parse()`-calls
  worden niet meer gebruikt, zodat een falende `field_validator` het antwoord en de usage niet
  verliest. Het schema dat naar de provider gaat is dat van `parse()` (`transform_schema`,
  `type_to_text_format_param`).
- **Validation retries**: `justai/tools/validation.py` bevat `RepairBudget`, de formatter en
  `ValidationRetryError`. Twee gebruikers: `Model.chat`/`prompt(validation_retries=N)` doet een
  repair call met de fouten (chat via de providerhistorie, prompt met vraag plus fout antwoord),
  en de `Agent` valideert elke tool call tegen een strict args-model (`create_model`,
  `extra='forbid'`) voordat de tool draait. Dat args-model is ook de bron van het tool-schema
  dat naar de provider gaat (`model_json_schema()`), zodat schema en validatie niet uit elkaar
  lopen. Kapotte tool-JSON uit `stream()` komt binnen als
  `ToolCallRequest.raw_arguments` en volgt dezelfde route. Drie soorten retries staan los van
  elkaar: transport in de SDK (`max_retries` op `Model`), rate limit en verbinding in de
  Agent-lus (`Agent(max_retries=...)`) en validatie (`validation_retries`).
- **Agent-stopregel**: elke fout die de run stopt (providerfout of uitgeputte validation
  retries) eindigt met een `error`-event plus `done`, en `AgentResult.error` zegt waarom.
- **Multimodaal** — PIL Image, URL of raw bytes → automatische base64-encoding.
- **Classificatie** — `Model.classify()` op System One-modellen. Eén wire-formaat
  (`POST /v1/systemone`) achter drie ingangen: TypeSafe, OpenRouter en elke zelf-gehoste
  server. Het formaat staat als losse functies in `systemone.py`, de HTTP-helft in
  `SystemOneMixin`, die `OpenRouterModel` als tweede basis meekrijgt. Die mixin heeft een
  eigen keep-alive `httpx.Client` en een eigen retry-lus op 429 en 529, niet omdat de
  provider-SDK's dat niet kunnen maar omdat er hier geen SDK tussen zit: latency is het
  hele punt van deze modelcategorie, en een TLS-handshake van 50-100 ms weegt zwaar tegen
  een antwoord van 400 ms. De timeout is daarom 30 seconden, niet de 120 van
  `DEFAULT_TIMEOUT`. Er is geen emulatie op gewone LLM's: die leveren geen gekalibreerde
  kansen, en een `classify()` die er stiekem een chat-call van maakt zou de enige reden om
  deze modellen te gebruiken weggooien. Andere modellen krijgen `NotImplementedError` uit
  `BaseModel`.
