---
date: 2026-09-27
type: feat
topic: validation-retries
artifact_contract: ce-implementation-plan/v1
execution: code
---

# Validation retries: validate, repair, retry

## Wat er gebouwd wordt

Justai krijgt één kleine laag die LLM-output valideert en bij een fout het model om een
reparatie vraagt. Twee gebruikers delen dezelfde code:

1. **Structured output.** `Model.chat()` en `Model.prompt()` krijgen `validation_retries`.
   Faalt de Pydantic-validatie van het antwoord, dan gaan de fouten in compacte vorm terug
   naar het model, met de vraag om een gecorrigeerd antwoord. Dat herhaalt zich maximaal
   `validation_retries` keer.
2. **Tool arguments in `Agent`.** Elke tool call wordt tegen de signatuur van de tool
   gevalideerd voordat de tool draait. Ongeldige arguments gaan als tool result met
   foutfeedback terug naar het model, dat de call dan opnieuw doet. De tool draait nooit
   met ongeldige arguments.

Malformed JSON, in structured output en in tool arguments, telt als een gewone
repareerbare validatiefout. Voor structured output geldt dat bij Anthropic en OpenAI
Responses; Gemini en de completions-familie parsen de JSON zelf en raisen, zie Uitgesteld.

```python
result = model.chat('...', response_format=Person, validation_retries=2)
agent = Agent('claude-sonnet-5', tools=[create_invoice], validation_retries=2)
```

## Voorwaarde

De tweak die de stopfouten van de Agent gelijktrekt moet eerst op main staan. Na die tweak
eindigt elke fout die de Agent-run stopt met een `error`-event plus `done`, en draagt
`AgentResult` een veld `error: str | None`. Deze feature sluit daarop aan: uitgeputte
validation retries in de Agent volgen dezelfde regel. Staat die tweak er niet, stop dan en
meld het.

## Gedrag

### Structured output

- `validation_retries` is een keyword-argument op `Model.chat()` en `Model.prompt()`,
  default `0`.
- `0` is exact het huidige gedrag: één validatie, geen repair call, bij een fout de
  originele Pydantic `ValidationError`.
- Bij `> 0` en een ongeldig antwoord volgt een repair call. Na uitputting volgt een
  `ValidationRetryError`.
- `validation_retries > 0` vereist een Pydantic-klasse als `response_format`. Een dict-schema
  of `list[X]` geeft een `ValueError`, want daar bestaat vandaag geen validatie.
- **Repair via `chat()`**: de repair call gaat via dezelfde provider-`chat()` met alleen de
  feedback als prompt. De providerhistorie draagt het origineel en het foute antwoord.
- **Repair via `prompt()`**: `prompt()` is stateless. De repair prompt bevat het originele
  prompt, het foute antwoord en de feedback. Images gaan alleen mee met de eerste call.
- **Cache (`prompt(cached=True)`)**: alleen het uiteindelijke geldige resultaat wordt
  opgeslagen, onder de key van het originele prompt. Een cache hit wordt gewoon
  gevalideerd, zonder repair.
- **Tokentellers**: `input_token_count` en `output_token_count` tellen alle attempts op. De
  repair calls zijn gefactureerd. Ook bij een exception blijven de opgetelde tellers staan.

### Tool arguments

- `Agent(validation_retries=2)` is de default. Validatie staat altijd aan; `0` betekent dat de
  eerste ongeldige call de run stopt.
- Per tool wordt bij registratie één args-model gebouwd met `pydantic.create_model`:
  - uit de signatuur van een callable, zonder `self` en `ctx`;
  - of uit de `params`-dict van een tool-object met `get_tools()`. Defaults komen uit de
    signatuur van de bijbehorende callable, zodat een weggelaten optioneel argument
    (`fetch_url(url, raw=False)`) geen fout geeft.
  - Een parameter zonder annotatie wordt `Any`. `extra='forbid'`, zodat verzonnen arguments
    een fout geven in plaats van een `TypeError` bij de aanroep.
  - Een parameter die zelf een Pydantic-model is (`args: InvoiceArgs`) valideert genest en
    komt als instantie binnen in de tool.
- De tool wordt aangeroepen met de gevalideerde waarden, niet met de rauwe dict. `"12"` voor
  een `int` komt dus als `12` binnen (Pydantic lax mode).
- Bij een ongeldige call:
  - de tool draait niet;
  - er komt geen `AuditEntry` (er is niets uitgevoerd);
  - het model krijgt via het bestaande `format_tool_result` met de `tool_call_id` van die
    call de feedback;
  - de Agent yieldt een `status`-event `Invalid arguments for <tool>, retry n/N`.
- De retry-teller telt per toolnaam en reset bij een geslaagde validatie van die tool.
  Meerdere tool calls in dezelfde beurt hebben elk hun eigen teller. De ene tool kan dus de
  teller van de andere niet opmaken.
- Na uitputting stopt de run volgens de regel uit de voorwaarde: `error`-event plus `done`,
  met `AgentResult.error` gevuld.
- Een onbekende toolnaam houdt het huidige gedrag: een foutmelding als tool result, zonder
  teller.

### Malformed JSON in tool arguments

- `ToolCallRequest` krijgt `raw_arguments: str | None = None`.
- Een gedeelde helper `parse_tool_arguments(raw) -> tuple[dict, str | None]` in
  `basemodel.py` geeft bij geldige JSON `(dict, None)` en bij ongeldige JSON `({}, raw)`.
- De drie `stream()`-parsers gebruiken die helper in plaats van een kale `json.loads`, zodat
  een kapotte tool call de stream niet meer laat crashen.
- De Agent valideert bij `raw_arguments` de rauwe string met `model_validate_json`. Pydantic
  levert dan een gewone `json_invalid`-fout, die dezelfde route volgt als elke andere fout.
- `format_assistant_message` stuurt de lege dict terug naar de provider. Die is geldig voor
  elk tool-protocol.

## De gedeelde primitive

Nieuw bestand `justai/tools/validation.py`, rond de 60 regels.

```python
class ValidationRetryError(ValueError):
    """Validation still failed after the allowed repair attempts."""
    attempts: int                # aantal validaties, eerste attempt inbegrepen
    last_error: ValidationError  # ook als __cause__
    raw: Any                     # laatste rauwe antwoord of arguments

def format_validation_errors(err: ValidationError) -> str: ...

class RepairBudget:
    def __init__(self, validate: Callable[[Any], Any], max_retries: int, label: str): ...
    def check(self, raw) -> tuple[Any, str | None]: ...
```

- `check()` valideert `raw`:
  - Geldig: `(value, None)`.
  - Ongeldig met budget over: `(None, feedback)`, plus een DEBUG-log
    `<label> validation failed, retry n/N`.
  - Ongeldig zonder budget: `ValidationRetryError`.
- `validate` is een callable die een Pydantic `ValidationError` raist. Andere exceptions
  gaan ongewijzigd door; dat zijn bugs, geen repareerbare output.
- `check()` bepaalt zelf niet hoe de volgende model-call gaat. Dat is het enige wat elke
  gebruiker zelf regelt:
  - `Model` in een `while`-lus rond de provider-call;
  - de `Agent` via zijn bestaande iteratielus.
  Tellen, begrenzen, formatteren, loggen en de exception staan zo op één plek.
- De twee feedbackteksten staan ook in dit bestand, zodat er maar één formatter is:

```text
The previous response failed validation.

Validation errors:
- name: Value error, Must contain first and last name

Return a corrected response that satisfies the required schema.
Do not explain the correction.
```

```text
Tool call `create_invoice` could not be executed because its arguments failed validation.

Validation errors:
- customer_id: Input should be a valid integer, unable to parse string as an integer
- amount: Input should be greater than 0

Correct the tool arguments and call the tool again.
```

- **Formatter-regels:**
  - Per fout één regel `- <pad>: <msg>`, met het pad als `loc` samengevoegd met punten
    (`items.2.quantity`, `customer.email`).
  - Een lege `loc` wordt `(root)`.
  - Geen `input`-waarden, geen `url`, geen schema, geen traceback. Zo lekt er niets van de
    payload in logs of feedback.
  - Maximaal 20 regels, daarna `- ... and N more`.
- **Logging:** via `logging.getLogger(__name__)`, net als de rest van de library. Er gaat
  nooit een payload in een logregel.
- **Async:** er is geen async variant nodig. De async API's van `Model` streamen tekst
  zonder `response_format`. De Agent is async, maar valideert synchroon tussen twee
  stream-calls in.

## Providers geven het antwoord terug in plaats van zelf te valideren

De SDK's van Anthropic en OpenAI valideren in hun `parse()`-calls zelf tegen de
Pydantic-klasse. Een `field_validator` die faalt, raist daardoor binnen de provider:

- nog vóór `record_usage`;
- zonder het rauwe antwoord;
- zonder dat het antwoord in de historie komt;
- bij OpenAI bovendien verpakt als `GeneralException`.

De repair-laag krijgt daar geen bruikbare fout te zien. Daarom verandert het contract:

> Bij een Pydantic-klasse als `response_format` geeft de provider de rauwe JSON-tekst terug.
> `Model` valideert centraal via het bestaande `_to_pydantic`.

| Provider | Nu | Wordt |
|---|---|---|
| Anthropic structured (`anthropic_models.py`, `_completion_with_structured_output`) | `client.messages.parse(output_format=cls)` | `client.messages.create(...)` met als format `transform_schema(cls.model_json_schema())` uit `anthropic.lib._parse._transform`, precies wat `parse` vandaag stuurt (`parse` vervangt het zelfgebouwde schema); `chat()` geeft `response_str` terug zonder `json.loads` |
| Anthropic structured, historie | assistant-antwoord ontbreekt in `self.messages` | assistant-tekst wordt aan `self.messages` toegevoegd, net als in het legacy-pad |
| Anthropic legacy-pad (`_parse_json_legacy`) | raist `JSONDecodeError` | bij een Pydantic-format: bij een parsefout de rauwe tekst teruggeven |
| OpenAI Responses (`openai_responses.py`, `prompt`) | `_responses_parse(text_format=cls)`, `output_parsed` | `_responses_create(text={'format': type_to_text_format_param(cls)})`, `output_text` |
| Google (`google_models.py`, `convert_to_justai_response`) | `response.parsed` of `_parse_gemini_json` | ongewijzigd, zie "Wat nog niet geverifieerd is" |
| Completions-familie | `response_format` alleen bij `openai.com`, geeft dict | ongewijzigd; `_to_pydantic` valideert de dict al centraal |

Het schema dat naar de provider gaat, blijft identiek. Alleen de validatie verhuist naar
`Model`. Wie vandaag `model.chat(..., response_format=Person)` doet, krijgt nog steeds een
`Person` of een `ValidationError`. Bij OpenAI is dat nu een `ValidationError` in plaats van
een `GeneralException`.

## Scheiding van retries

| Soort | Waar | Parameter |
|---|---|---|
| Transport, in de SDK | provider-clients | `max_retries` op `Model(...)` |
| Rate limit en verbinding, in de Agent | `Agent.run`-lus | `Agent(max_retries=...)` |
| Validation/repair | `RepairBudget` | `validation_retries` |

Er is geen gedeelde teller. Een rate-limit-retry binnen een repair call telt niet als
validation retry, en andersom.

## Bestanden

| Bestand | Nieuw | Wat erin komt |
|---|---|---|
| `justai/tools/validation.py` | ja | `ValidationRetryError`, `format_validation_errors`, `RepairBudget`, de twee feedbackteksten |
| `justai/model/model.py` | nee | `validation_retries` op `chat()` en `prompt()`, repairlus, opgetelde tellers |
| `justai/models/anthropic_models.py` | nee | `create` in plaats van `parse`, rauwe tekst, historie-fix, `parse_tool_arguments` in `stream()` |
| `justai/models/openai_responses.py` | nee | `create` met `type_to_text_format_param`, `output_text`, `parse_tool_arguments` in `stream()` |
| `justai/models/openai_completions.py` | nee | `parse_tool_arguments` in `stream()` |
| `justai/models/basemodel.py` | nee | `ToolCallRequest.raw_arguments`, `parse_tool_arguments()` |
| `justai/agent/agent.py` | nee | args-model per tool, `validation_retries`, validatie voor uitvoering |
| `justai/__init__.py` | nee | `ValidationRetryError` exporteren |
| `tests/test_validation.py` | ja | primitive en formatter |
| `tests/test_validation_retries.py` | ja | `Model.chat`/`prompt` met een nep-provider |
| `tests/test_structured_passthrough.py` | ja | providers geven rauwe tekst terug, met gemockte SDK-clients |
| `tests/test_agent_validation.py` | ja | Agent met een nep-`stream()` |
| `README.md`, `docs/architecture.md`, `CLAUDE.md`, `AGENTS.md` | nee | korte vermelding plus de twee voorbeelden |

## Execution table

| # | Task | Touches | Depends on |
|---|------|---------|------------|
| 01 | Primitive: `ValidationRetryError`, formatter, `RepairBudget`, feedbackteksten | `justai/tools/validation.py`, `tests/test_validation.py` | - |
| 02 | Providers geven rauwe tekst terug bij een Pydantic-format, Anthropic-historie-fix | `justai/models/anthropic_models.py`, `justai/models/openai_responses.py`, `tests/test_structured_passthrough.py` | - |
| 03 | `Model.chat`/`prompt` met `validation_retries` | `justai/model/model.py`, `justai/__init__.py`, `tests/test_validation_retries.py` | 01, 02 |
| 04 | `parse_tool_arguments` en `raw_arguments` in de drie `stream()`-parsers | `justai/models/basemodel.py`, `justai/models/anthropic_models.py`, `justai/models/openai_responses.py`, `justai/models/openai_completions.py` | 02 (zelfde bestanden) |
| 05 | Agent: args-model, validatie voor uitvoering, `validation_retries` | `justai/agent/agent.py`, `tests/test_agent_validation.py` | 01, 04 |
| 06 | Documentatie, inclusief de stopregel van de Agent (error plus done, `AgentResult.error`) in `docs/architecture.md` | `README.md`, `docs/architecture.md`, `CLAUDE.md`, `AGENTS.md` | 03, 05 |

01 en 02 kunnen parallel. 03 en 04 kunnen parallel na 02. Alleen de hoofdsessie commit.

## Tests

Alle tests draaien zonder netwerk en zonder API-keys. Structured-output-tests vervangen
`model.model.chat`/`prompt` door een nep die een lijst antwoorden afloopt en elke call
vastlegt. Agent-tests vervangen `agent.model.model.stream` door een async generator die per
iteratie een voorgeprogrammeerde `StreamChunk`-reeks levert. `format_assistant_message` en
`format_tool_result` komen van een echt Anthropic-model met een dummy key. Draaien met
`uv run pytest tests/test_validation*.py tests/test_structured_passthrough.py
tests/test_agent_validation.py tests/test_agent.py`.

### Task 01, `tests/test_validation.py`

| Test | Bewijst |
|---|---|
| `test_format_single_error` | `- name: Value error, Must contain first and last name` |
| `test_format_nested_path` | `customer.email` voor een genest model |
| `test_format_list_path` | `items.2.quantity` voor een veld in een lijst |
| `test_format_root_error` | lege `loc` wordt `(root)` |
| `test_format_omits_input` | de ongeldige waarde staat niet in de output |
| `test_format_caps_at_20` | 25 fouten geven 20 regels plus `- ... and 5 more` |
| `test_check_valid_returns_value` | `(value, None)` |
| `test_check_invalid_returns_feedback` | `(None, feedback)` met alle foutregels |
| `test_check_exhausted_raises` | na `max_retries` keer ongeldig volgt `ValidationRetryError` met `attempts == max_retries + 1`, `last_error` en `raw` |
| `test_check_zero_retries_raises_at_once` | `max_retries=0` raist bij de eerste fout |
| `test_check_other_exception_passes_through` | een `KeyError` uit `validate` raist ongewijzigd en verbruikt geen budget |
| `test_retry_error_is_value_error` | `except ValueError` vangt hem |
| `test_debug_log_has_no_payload` | de logregel bevat label en `retry 1/2`, niet de waarde |

### Task 02, `tests/test_structured_passthrough.py`

| Test | Bewijst |
|---|---|
| `test_anthropic_structured_uses_create_not_parse` | met een Pydantic-format wordt `messages.create` aangeroepen met het `output_config`-schema |
| `test_anthropic_structured_returns_raw_text` | een antwoord dat de `field_validator` breekt komt als tekst terug, zonder raise, en `last_usage` is gezet |
| `test_anthropic_structured_appends_assistant` | na `chat()` staat het assistant-antwoord in `self.messages` |
| `test_anthropic_legacy_returns_raw_on_bad_json` | legacy-pad plus Pydantic-format plus onleesbare tekst geeft de tekst terug |
| `test_anthropic_legacy_return_json_still_raises` | `return_json=True` zonder `response_format` raist nog steeds op onleesbare JSON |
| `test_openai_pydantic_uses_create_with_strict_format` | `_responses_create` krijgt `text.format` van `type_to_text_format_param`, `_responses_parse` wordt niet aangeroepen |
| `test_openai_pydantic_returns_output_text` | het rauwe `output_text` komt terug en `last_response_id` is gezet |

### Task 03, `tests/test_validation_retries.py`

| Test | Bewijst |
|---|---|
| `test_valid_first_time_one_call` | geen retry, één provider-call |
| `test_invalid_then_valid` | twee calls, eindresultaat is een `Person` |
| `test_repair_prompt_carries_feedback` | de tweede `chat`-call krijgt de feedbacktekst als prompt |
| `test_multiple_errors_all_in_feedback` | beide foutregels staan in de repair prompt |
| `test_exhausted_raises_retry_error` | na `1 + validation_retries` calls volgt `ValidationRetryError` en geen extra call |
| `test_zero_retries_raises_validation_error` | `validation_retries=0` geeft de originele `ValidationError` na één call |
| `test_default_is_zero` | zonder argument hetzelfde als `0` |
| `test_nested_model_path` | `customer.email` in de feedback |
| `test_list_field_path` | `items.2.quantity` in de feedback |
| `test_malformed_json_is_repairable` | `'{"name": '` als eerste antwoord leidt tot een repair call met een `Invalid JSON`-regel |
| `test_prompt_repair_includes_original_and_answer` | de repair via `prompt()` bevat het originele prompt, het foute antwoord en de feedback |
| `test_prompt_cache_stores_only_final` | `cache_save` wordt één keer aangeroepen, met het geldige resultaat |
| `test_token_counts_summed` | tellers zijn de som van alle attempts, ook na `ValidationRetryError` |
| `test_retries_require_pydantic_format` | `validation_retries=1` met een dict-schema geeft `ValueError` |

### Task 04, uitbreiding van `tests/test_validation.py`

| Test | Bewijst |
|---|---|
| `test_parse_tool_arguments_valid` | `('{"a": 1}')` geeft `({'a': 1}, None)` |
| `test_parse_tool_arguments_empty` | `''` geeft `({}, None)`, het huidige gedrag |
| `test_parse_tool_arguments_malformed` | `'{"a": '` geeft `({}, '{"a": ')` zonder raise |

### Task 05, `tests/test_agent_validation.py`

| Test | Bewijst |
|---|---|
| `test_valid_args_run_tool_once` | tool precies één keer aangeroepen |
| `test_invalid_then_valid` | eerste call voert de tool niet uit, het tool result bevat de feedback, de tweede call voert hem één keer uit |
| `test_coerced_values_reach_tool` | `"12"` voor een `int` komt als `12` binnen |
| `test_pydantic_param_arrives_as_instance` | `args: InvoiceArgs` komt binnen als `InvoiceArgs` |
| `test_extra_argument_rejected` | een verzonnen argument is een validatiefout, geen `TypeError` |
| `test_always_invalid_never_runs` | tool nooit aangeroepen, run eindigt met `error` en `done`, `AgentResult.error` gevuld, precies `1 + validation_retries` stream-calls |
| `test_counters_per_tool` | tool A faalt twee keer en tool B slaagt ertussen; B draait, A's teller loopt door |
| `test_counter_resets_after_success` | na een geslaagde call van A begint A's teller opnieuw |
| `test_two_invalid_calls_same_turn` | beide calls krijgen hun eigen feedback met hun eigen `tool_call_id` |
| `test_malformed_json_args_repairable` | `raw_arguments` leidt tot feedback met `Invalid JSON`, niet tot een crash |
| `test_ctx_excluded_from_args_model` | een `@agent.tool` met `ctx` valideert zonder `ctx` in de arguments |
| `test_get_tools_object_validated` | een tool-object uit `get_tools()` krijgt ook een args-model |
| `test_no_audit_entry_for_rejected_call` | afgewezen calls staan niet in `AgentResult.audit` |

### Regressie

- De bestaande tests blijven groen: `tests/test_agent.py`, `test_prompt_caching.py`,
  `test_cache.py`, `test_google_json.py`, `test_usage_on_failure.py`,
  `test_anthropic_thinking_blocks.py` en `test_effort.py`.
- `test_anthropic_thinking_blocks.py`, `test_prompt_caching.py`, `test_usage_on_failure.py`
  en `test_effort.py` mocken `client.messages.parse`. Die mocks gaan naar
  `client.messages.create`; verder verandert er niets aan die tests.
- `test_agent_execute_tool_error` test een tool die zelf raist, geen verkeerde arguments,
  en blijft ongewijzigd.

## Geverifieerd tegen de SDK

- **Google** `response.parsed`: google-genai vangt een `ValidationError` bij het parsen
  zelf af, `parsed` is dan `None` en `_parse_gemini_json` levert een dict op die
  `_to_pydantic` centraal valideert. Een `field_validator`-fout is daar dus al
  repareerbaar.
- **Privémodules**: `type_to_text_format_param` (`openai.lib._parsing._responses`) en
  `transform_schema` (`anthropic.lib._parse._transform`) zijn wat de SDK's intern in
  `parse` gebruiken. Pin beide in een test, zodat een SDK-upgrade die ze verplaatst direct
  faalt.

## Buiten scope

Semantic of LLM-validators, partial structured output, streaming van gedeeltelijk
gevalideerde objecten, context-aware validators, een dependency op `instructor` of
PydanticAI, een nieuwe agent-abstractie, provider-specifieke validation-API's.

## Uitgesteld

- `validation_retries` voor niet-Pydantic formats (`list[X]`, dict-schema's) via
  `TypeAdapter`. Trigger: de eerste caller die dat nodig heeft.
- Validatie van automatic function calling in `Model.prompt`/`chat` via `Model.add_tool`
  (`openai_responses.py` roept `function(*args.values())` ongevalideerd aan). Trigger: een
  bug of feature die dat pad raakt.
- Malformed JSON repareerbaar maken bij Gemini (`_parse_gemini_json` raist
  `GeneralException`) en de completions-familie (`json.loads` in
  `openai_completions.py`). Trigger: de eerste melding van een niet-gerepareerde
  JSON-fout bij een van die providers.
