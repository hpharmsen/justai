---
title: 'feat: prompt caching breakpoints en cachestatistieken'
type: feat
status: active
date: 2026-09-08
---

# feat: prompt caching breakpoints en cachestatistieken

## Doel

Anthropic prompt caching laten werken op de plekken waar het geld oplevert, en de hit rate meetbaar maken. Vandaag zet justai precies één breakpoint, achter een handmatig gezette `cached_prompt`, en alleen in het niet-streaming pad. De agent-loop en multi-turn gesprekken cachen niets.

## Aanleiding

Anthropic meldt een lage cache hit rate op het API-verkeer van Harmsen.nl AI Consultancy en schat de besparing op tot 56%. Claude Code is uitgesloten van die meting, dus het gaat om verkeer dat via justai loopt.

## Probleemanalyse

**1. `AnthropicModel.stream()` cachet niets.** Regel 575-580 van `justai/models/anthropic_models.py` bouwt de request met de systeemprompt als kale string en zonder enige `cache_control`:

```python
api_params = {'model': ..., 'messages': api_messages, 'system': system_message, **self.api_params}
```

`justai/agent/agent.py:293` roept dit aan in een lus die per iteratie de volledige, groeiende geschiedenis plus alle tool-resultaten opnieuw meestuurt (regels 344 en 359). Elke iteratie betaalt de hele prefix opnieuw tegen vol tarief. Bij een agent van vijftien iteraties gaan de systeemprompt en de tool-definities vijftien keer volledig over de lijn in plaats van één keer schrijven en veertien keer lezen tegen 0,1x. Dit is de duurste workload in de package en de enige zonder cache.

**2. Gespreksgeschiedenis krijgt nooit een breakpoint.** `cached_system_message()` (regel 657) zet er één achter `cached_prompt` in de systeemblokken. De messages-array blijft onaangeraakt, dus in `chat()` betaalt elke beurt de hele voorgaande conversatie opnieuw.

**3. Tool-definities.** Die renderen vóór `system`, dus in `completion()` liften ze mee op het system-breakpoint. In `stream()` niet, want daar staat geen breakpoint.

**4. Caching staat standaard uit.** Zonder expliciete `model.cached_prompt = ...` gebeurt er niets.

**5. Geen 1-uurs TTL.** Alleen `{'type': 'ephemeral'}`, dus vijf minuten. Verkeer met gaten van meer dan vijf minuten verliest de cache steeds.

**6. Cachestatistieken zijn onzichtbaar.** Regel 292 leest de cachevelden alleen uit als `cached_prompt` gezet is, `stream()` telt ze helemaal niet, en OpenAI en Gemini rapporteren hun cached tokens nergens. Je kunt met justai dus niet meten of caching werkt, wat precies de meting is die de Anthropic-mail vraagt.

## Aanpak

Eén pure helper in `anthropic_models.py` die `(system, messages)` in en uit gaat, en op maximaal twee plekken een breakpoint zet:

1. Op het laatste systeemblok. Dat cachet tools plus systeemprompt in één keer, omdat de rendervolgorde `tools` → `system` → `messages` is.
2. Op het laatste content-blok van de laatste message, maar alleen als er meer dan één message is.

Die tweede regel is de kern van het kostenmodel. Een cache write kost 1,25x. Bij een echte one-shot call (`prompt()` reset de messages en stuurt één user-message) zou een breakpoint op de geschiedenis 25% extra kosten zonder ooit gelezen te worden. Vanaf de tweede message is er per definitie iets te hergebruiken en verdient de write zich terug bij de eerstvolgende beurt.

De helper werkt copy-on-write. De agent hergebruikt zijn `messages`-lijst tussen iteraties; als we die dicts muteren dan verschilt de bytes-prefix bij de volgende request van wat er gecached is en mist de cache alsnog, stil.

Daarnaast: cachetellers naar `BaseModel` verhuizen zodat elke provider ze heeft, `record_usage()` uitbreiden met twee optionele keyword-argumenten, en de cached-token-velden van OpenAI en Gemini doorgeven.

**Waarom deze aanpak:** het zijn twee breakpoints van de vier die toegestaan zijn, één helper, geen nieuwe abstractielaag en geen configuratie die de gebruiker moet begrijpen om winst te pakken. Standaard aan, met een uitschakelaar voor wie het niet wil.

**Wat we niet doen:**

- Geen expliciete Gemini CachedContent-API. Gemini doet impliciete caching, dat volstaat en de expliciete variant is een heel ander levenscyclusmodel.
- Geen automatische cache pre-warming.
- Geen breakpoint-strategie met meer dan twee punten. De 20-blokken lookback kan dat later nodig maken, zie risico 4.

## Executietabel

| # | Taak | Raakt | Hangt af van |
|---|------|-------|--------------|
| 01 | Cache-helper plus breakpoints in `completion()`, `stream()` en het structured-output pad | `justai/models/anthropic_models.py`, `tests/test_prompt_caching.py` | - |
| 02 | Cachetellers naar `BaseModel`, `record_usage()` uitbreiden, `Model`-properties vereenvoudigen | `justai/models/basemodel.py`, `justai/model/model.py`, `tests/test_cache_stats.py` | - |
| 03 | Cached tokens rapporteren voor OpenAI en Gemini | `justai/models/openai_responses.py`, `justai/models/openai_completions.py`, `justai/models/google_models.py` | 02 |
| 04 | Anthropic cachestatistieken altijd registreren, ook in `stream()` | `justai/models/anthropic_models.py` | 01, 02 |
| 05 | `cache_ttl` en `prompt_cache=False` schakelaar | `justai/models/anthropic_models.py`, `examples/prompt_caching.py` | 04 |
| 06 | Documentatie | `README.md`, `docs/architecture.md`, `CLAUDE.md` | 01, 05 |

Taken 01 en 02 kunnen tegelijk. Daarna 03 en 04 tegelijk. 05 en 06 zijn de staart. Taken 01, 04 en 05 raken allemaal `anthropic_models.py` en staan daarom in een keten.

## Tests (TDD-first)

### `tests/test_prompt_caching.py` (nieuw, taak 01)

Kopieer de `isolate_cache_and_warnings` fixture uit `tests/test_effort.py:24-31`. Alle tests draaien zonder API-key, tegen de pure helper plus een gemockte client.

```python
def test_system_breakpoint_on_last_block():
    system, messages = apply_cache_control('lange systeemprompt', [{'role': 'user', 'content': 'hi'}])
    assert isinstance(system, list)
    assert system[-1]['cache_control'] == {'type': 'ephemeral'}

def test_no_history_breakpoint_on_single_message():
    _, messages = apply_cache_control('sys', [{'role': 'user', 'content': 'hi'}])
    assert 'cache_control' not in str(messages)

def test_history_breakpoint_from_second_message():
    msgs = [
        {'role': 'user', 'content': 'een'},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'twee'}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'drie'}]},
    ]
    _, out = apply_cache_control('sys', msgs)
    assert out[-1]['content'][-1]['cache_control'] == {'type': 'ephemeral'}

def test_input_is_not_mutated():
    # De agent hergebruikt zijn messages-lijst tussen iteraties. Muteren breekt de
    # volgende prefix-match, stil en zonder foutmelding.
    original = [{'role': 'user', 'content': 'een'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'twee'}]}]
    snapshot = copy.deepcopy(original)
    apply_cache_control('sys', original)
    assert original == snapshot

def test_existing_breakpoint_is_not_duplicated():
    # cached_system_message() zet er zelf al een op het laatste blok
    model = AnthropicModel('claude-sonnet-4-6', {'ANTHROPIC_API_KEY': 'test'})
    model.cached_prompt = 'lange tekst'
    system, _ = apply_cache_control(model.cached_system_message(), [{'role': 'user', 'content': 'hi'}])
    assert sum('cache_control' in b for b in system) == 1

def test_ttl_1h(): ...           # {'type': 'ephemeral', 'ttl': '1h'}
def test_disabled_returns_input_unchanged(): ...
def test_never_more_than_four_breakpoints(): ...
def test_empty_system_is_left_alone(): ...   # een leeg tekstblok geeft een 400
```

Plus twee end-to-end tests met een gemockte client, in de stijl van `_mock_anthropic_response` uit `test_effort.py:34`:

```python
def test_completion_sends_breakpoints(monkeypatch): ...   # inspecteer client.messages.create kwargs
def test_stream_sends_breakpoints(monkeypatch): ...       # system is een lijst met cache_control
```

### `tests/test_cache_stats.py` (nieuw, taak 02 en 03)

```python
def test_basemodel_defaults_to_zero(): ...
def test_record_usage_sets_cache_fields(): ...
def test_anthropic_reports_cache_tokens_without_cached_prompt(): ...  # regressie op de huidige if-gate
def test_openai_responses_reports_cached_tokens(): ...   # usage.input_tokens_details.cached_tokens
def test_openai_completions_reports_cached_tokens(): ... # usage.prompt_tokens_details.cached_tokens
def test_gemini_reports_cached_tokens(): ...             # usage_metadata.cached_content_token_count
def test_missing_usage_details_do_not_raise(): ...       # oudere SDK-responses hebben het veld niet
```

### Integratietest (opt-in, echte key)

```python
@pytest.mark.skipif(not os.getenv('ANTHROPIC_API_KEY'), reason='geen API key')
def test_second_call_reads_cache():
    model = Model('claude-sonnet-4-6')
    model.cached_prompt = LONG_TEXT     # ruim boven 1024 tokens
    model.prompt('Vraag een', cached=False)   # cached=False omzeilt justais eigen lokale cache
    model.prompt('Vraag twee', cached=False)
    assert model.cache_read_input_tokens > 0
```

**Verificatie:**

```bash
uv run pytest tests/test_prompt_caching.py tests/test_cache_stats.py
uv run pytest tests/test_agent.py tests/test_effort.py tests/test_anthropic_thinking_blocks.py   # regressie
```

## Implementatie

### Taak 01: helper plus breakpoints

Nieuw in `justai/models/anthropic_models.py`, boven de klasse:

```python
def _breakpoint(ttl: str | None) -> dict:
    """cache_control blok. De default TTL is 5 minuten en wordt niet meegestuurd."""
    return {'type': 'ephemeral', 'ttl': ttl} if ttl and ttl != '5m' else {'type': 'ephemeral'}


def apply_cache_control(system, messages: list[dict], ttl: str | None = None, enabled: bool = True):
    """Zet cache-breakpoints zonder de input te muteren.

    Twee punten, van de vier die de API toestaat:
      1. laatste systeemblok, dat cachet tools plus systeemprompt (rendervolgorde
         is tools, system, messages)
      2. laatste content-blok van de laatste message, maar alleen vanaf de tweede
         message: een enkele message heeft nog niets om te hergebruiken en zou
         alleen de 1,25x write-premie kosten

    Copy-on-write: de agent hergebruikt zijn messages tussen iteraties, en een
    gemuteerde dict laat de volgende prefix-match stil mislukken.
    """
```

Aanroepen op vier plekken:

- `completion()` regel 479, waar nu `api_params['system'] = system_message` staat, samen met `api_params['messages']`
- `completion()` streaming-tak, regel 424-431
- `_completion_with_structured_output()` regel 352-358
- `stream()` regel 575-580

De helper accepteert zowel een string als de lijst uit `cached_system_message()`, zodat het bestaande `cached_prompt`-pad ongewijzigd blijft werken.

### Taak 02: cachetellers naar BaseModel

In `justai/models/basemodel.py` bij `__init__` (rond regel 194, naast `last_usage`):

```python
self.cache_creation_input_tokens = 0
self.cache_read_input_tokens = 0
```

En `record_usage` (regel 219) uitbreiden met keyword-only argumenten, zodat de vijf bestaande aanroepers ongewijzigd blijven:

```python
def record_usage(self, input_tokens, output_tokens, *, cache_creation_tokens=0, cache_read_tokens=0) -> None:
```

In `justai/model/model.py` vervallen de `hasattr`-checks in de properties op regel 116-126: de attributen bestaan nu altijd. Nul is voor een provider die niets rapporteert een eerlijker antwoord dan een `AttributeError`. De `cached_prompt`-property blijft wel gaten schieten, want dat is een echt capability-verschil (zie `supports_cached_prompts`).

### Taak 03: OpenAI en Gemini

| Provider | Veld | Aanroepsite |
|---|---|---|
| OpenAI Responses | `usage.input_tokens_details.cached_tokens` | `openai_responses.py:191` |
| OpenAI Completions | `usage.prompt_tokens_details.cached_tokens` | `openai_completions.py:175` |
| Gemini | `usage_metadata.cached_content_token_count` | `google_models.py:391` |

Alle drie defensief uitlezen met `getattr(..., 0) or 0`; oudere SDK-responses missen het veld. Deze providers cachen impliciet en server-side, dus alleen de read-teller vult zich, `cache_creation_input_tokens` blijft nul.

Let op het semantische verschil dat de documentatie moet noemen: bij OpenAI en Gemini zijn de cached tokens een deelverzameling van de input tokens, bij Anthropic staan ze er los van.

### Taak 04: Anthropic-statistieken altijd

`chat()` regel 292-296 verliest zijn `if self.cached_prompt`-gate en leest de velden altijd uit, via `record_usage(..., cache_creation_tokens=..., cache_read_tokens=...)`. In `stream()` komen ze binnen op het `message_start`-event (regel 605-607), naast `input_tokens`. Zonder deze taak is er geen enkele manier om te zien of taak 01 werkt in het agent-pad.

### Taak 05: TTL en uitschakelaar

Twee model-params, beide met een default die niemand hoeft te kennen:

- `cache_ttl='5m'`, alternatief `'1h'`. Een 1-uurs write kost 2x in plaats van 1,25x en verdient zich pas terug vanaf de derde request, dus dit blijft opt-in.
- `prompt_cache=True`. Op `False` gedraagt justai zich als vandaag. Ontsnappingsluik voor wie echt alleen one-shot calls doet en de write-premie niet wil.

Beide toevoegen aan `_NON_API_PARAMS` in `basemodel.py`, anders gaan ze mee de provider-API in.

`examples/prompt_caching.py` uitbreiden met een multi-turn voorbeeld dat de hit rate over drie beurten laat zien.

### Taak 06: documentatie

- `README.md` sectie "Prompt caching (Anthropic)" regel 120-125: beschrijf dat caching nu standaard aan staat, wat `cached_prompt` nog toevoegt, en documenteer `cache_ttl` en `prompt_cache`.
- `README.md`: cachetellers zijn nu ook voor OpenAI en Gemini beschikbaar, met de kanttekening over inclusief versus exclusief tellen.
- `docs/architecture.md` regel 74: "Prompt caching - Provider-specifiek (Anthropic native, andere via lokale cache)" klopt niet meer; herschrijf naar automatische breakpoints bij Anthropic plus doorgegeven statistieken elders.
- `CLAUDE.md`: één regel bij Key Features Architecture.

## Kritische bestanden

- `justai/models/anthropic_models.py` — helper plus vier aanroepplekken (regels 292, 352, 424, 479, 575, 605, 657)
- `justai/models/basemodel.py` — tellers en `record_usage` (regels 194, 219)
- `justai/model/model.py` — properties (regels 116-126)
- `justai/agent/agent.py` — alleen lezen, dit is de consument die de winst incasseert
- `justai/models/openai_responses.py`, `openai_completions.py`, `google_models.py` — één regel per bestand
- `tests/test_prompt_caching.py`, `tests/test_cache_stats.py` — nieuw

## Risico's en open vragen

1. **Minimale cachebare prefix verschilt per model.** 512 tokens voor Opus 5 en Fable 5, 1024 voor Sonnet 5, Sonnet 4.6 en Opus 4.8, 2048 voor Opus 4.7, 4096 voor Opus 4.6 en Haiku 4.5. Onder die grens cachet de API stil niet, zonder foutmelding en zonder kosten. Niets aan doen, wel documenteren, en niet schrikken van `cache_creation_input_tokens == 0` bij korte prompts.

2. **Write-premie bij echt eenmalig gebruik.** Een cache write kost 1,25x. Wie precies één call doet met een systeemprompt boven de drempel betaalt 25% extra op de input tokens van die ene call. De regel "geen history-breakpoint bij één message" dekt het grootste deel; voor de rest is `prompt_cache=False` het antwoord. In de praktijk hergebruikt vrijwel elke justai-gebruiker dezelfde systeemprompt over meerdere calls, dus dit is een randgeval.

3. **Mutatie van de messages van de aanroeper.** Als de helper de dicts van de agent aanpast, verschilt de prefix bij de volgende iteratie en mist de cache alsnog, zonder enig signaal. Dit is de belangrijkste correctheidsvalkuil van het hele plan, vandaar de expliciete test `test_input_is_not_mutated`.

4. **De 20-blokken lookback.** Een breakpoint kijkt maximaal twintig content-blokken terug om een eerdere cache-entry te vinden. Een agent-iteratie met veel parallelle tool-calls kan daar overheen gaan, waarna de keten stil breekt. Met twee breakpoints van de vier gebruikt is er ruimte voor een tussenliggend punt. Follow-up, pas doen als de meting laat zien dat het gebeurt.

5. **Verwarrende tokentelling.** Bij Anthropic sluit `input_tokens` de gecachte tokens uit, bij OpenAI en Gemini niet. `Model.last_token_count()` telt dus na deze wijziging bij Anthropic niet meer het volledige promptvolume. Dat was al zo, maar wordt zichtbaarder. Documenteren in de README, niet stilzwijgend corrigeren.

6. **Denkblokken en cache.** Thinking aan- of uitzetten invalideert de messages-cache maar niet tools plus system. Geen actie, wel goed om te weten bij het interpreteren van de eerste metingen.

7. **Providers die niets rapporteren** (GGUF, Perplexity, Reve, xAI) houden nul. Eerlijk, maar documenteer dat nul daar "niet gemeten" betekent en niet "geen cache".

## Definition of done

- [ ] `uv run pytest tests/test_prompt_caching.py tests/test_cache_stats.py` slaagt
- [ ] `uv run pytest tests/` geeft geen regressies, in het bijzonder `test_agent.py`, `test_effort.py` en `test_anthropic_thinking_blocks.py`
- [ ] Integratietest met echte key: tweede call heeft `cache_read_input_tokens > 0`
- [ ] Agent-run van minimaal vijf iteraties met een systeemprompt boven de drempel laat vanaf iteratie twee `cache_read_input_tokens > 0` zien
- [ ] `examples/prompt_caching.py` toont de hit rate over drie beurten
- [ ] README.md, docs/architecture.md en CLAUDE.md bijgewerkt
- [ ] Handmatig: `Model('gpt-5-mini')` en `Model('gemini-2.5-flash')` geven een `cache_read_input_tokens` terug in plaats van een AttributeError

## Succesmetriek

Meet vóór en na op dezelfde agent-run: totaal gefactureerde input tokens en `cache_read_input_tokens`. Verwachting bij een run van tien iteraties met een systeemprompt van circa 2000 tokens plus tool-definities: vanaf iteratie twee komt het leeuwendeel van de prefix uit de cache tegen 0,1x. De hit rate in het Anthropic-dashboard is de uiteindelijke controle, met een paar dagen vertraging.

## Follow-up

- [ ] Tussenliggend breakpoint als de 20-blokken lookback in de praktijk breekt (risico 4)
- [ ] Cache pre-warming met `max_tokens: 0` voor apps met verkeer in bursts
- [ ] Expliciete Gemini CachedContent voor grote vaste contexten
- [ ] Overweeg een `cache_stats()` helper op `Model` die de vier tellers plus een afgeleide hit rate teruggeeft
