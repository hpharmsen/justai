---
date: 2026-09-26
type: feat
topic: model-classify-system-one
requirements: docs/brainstorms/2026-09-26-classify-system-one-requirements.md
artifact_contract: ce-implementation-plan/v1
execution: code
---

# Model.classify: System One-modellen in justai

## Wat er gebouwd wordt

Justai krijgt `Model.classify()`. Een gebruiker legt een tekst voor aan een System One-model
en krijgt een Python dict terug met het gekozen antwoord, de confidence en de gekalibreerde
kansverdeling over de opties. Drie ingangen: Jev via OpenRouter, Jev native bij TypeSafe, en
elk zelf-gehost model dat `POST /v1/systemone` serveert. Elk ander model gooit
`NotImplementedError`.

Daarnaast een voorbeeldscript in `examples/`, met in de comments de andere modellen die
dezelfde vorm serveren, en bijgewerkte documentatie in `README.md`, `CLAUDE.md`, `AGENTS.md`
en `docs/architecture.md`.

## Het wire-formaat

Geverifieerd tegen de leverancierdocumentatie op 26 september 2026.

Request, identiek bij beide providers, alle drie de velden verplicht:

```json
{"model": "jev-latest", "state": <str|dict|list>, "questions": {"<naam>": {...}}}
```

Response:

```json
{"id": "...", "model": "...",
 "answers": {"<naam>": {...}},
 "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992}}
```

De `usage`-veldnamen zijn bij OpenRouter en TypeSafe identiek. Dat is niet vanzelfsprekend,
want OpenRouters OpenAI-compatible endpoints gebruiken `prompt_tokens` en `completion_tokens`.
Eén parser volstaat dus voor beide. `cost` wordt genegeerd, zie Buiten scope.

| Type | `criteria` | Antwoordvelden | Grens |
|---|---|---|---|
| `choice` | dict van optie naar beschrijving | `type`, `choice`, `probabilities`, `confidence` | max 255 opties |
| `score` | geordende lijst van niveaubeschrijvingen | `type`, `score`, `legend`, `probabilities`, `confidence` | 2 tot en met 10 niveaus |
| `noul` | afwezig | `type`, `noul` | geen |

`legend` mapt het niveaunummer terug naar de beschrijving: `{"0": "Calm", "1": "Frustrated",
"2": "Very angry"}`.

Endpoints. Beide vragen alleen `Authorization: Bearer` en `Content-Type`, geen `HTTP-Referer`
of `X-Title`:

| Ingang | URL | Key |
|---|---|---|
| `openrouter/typesafe/jev-*` | `https://openrouter.ai/api/alpha/decisions` | `OPENROUTER_API_KEY` |
| `jev*` | `https://api.typesafe.ai/v1/systemone` | `TYPESAFE_API_KEY` |
| `systemone/<naam>` | `<base_url>/v1/systemone` | `SYSTEMONE_API_KEY`, optioneel |

Statuscodes, met de justai-exception ernaast:

| Code | Betekenis volgens de documentatie | Exception |
|---|---|---|
| 401 | "Missing or invalid API key" | `AuthorizationException` |
| 403 | niet gedocumenteerd, wel gebruikelijk | `AuthorizationException` |
| 422 | "The request body failed validation" | `BadRequestException` |
| 429 | "You have exceeded your rate limit. Back off and retry after a short delay." | `RatelimitException` na de retries |
| 529 | "TypeSafe is temporarily overloaded. Retry after a short delay." | `ModelOverloadException` na de retries |
| overige 5xx | niet gedocumenteerd | `ConnectionException` |

Validatie faalt met 422, niet met 400. 400 wordt voor de zekerheid ook op
`BadRequestException` gemapt, want OpenRouter documenteert zijn alpha-endpoint niet op dit
punt.

## Gecorrigeerde aannames uit het requirementsdocument

- **R11 klopt niet over `last_response_time`.** `record_usage()` in `basemodel.py:229` zet
  `last_usage` en de twee cachetellers, niets meer. `last_response_time` wordt gezet door
  `Model.prompt` zelf (`model.py:218`). `Model.classify` klokt dus zijn eigen call.
- **R10 vraagt meer dan een vlag.** `CacheDB.write` bindt `value` als sqlite TEXT
  (`cache.py:145-157`), dus een dict gaat er niet in. Classify serialiseert naar JSON voor
  het schrijven en leest terug met `json.loads`, precies zoals `Model.prompt` dat doet voor
  structured results (`model.py:16-19`, `model.py:176`, `model.py:203`).
- **`model` is een verplicht requestveld** bij beide providers. Het requirementsdocument
  noemt het nergens. De requestbuilder stuurt de naam zonder justai-prefix mee, dus
  `systemone/kev-3b` gaat als `kev-3b` de deur uit.
- **`noul` accepteert een optionele `criteria`** met een `"true"`- en een `"false"`-beschrijving.
  R2 laat `noul` volgen uit een ontbrekende `options`, waarmee die beschrijvingen onbereikbaar
  worden. Dat blijft zo in v1, zie Buiten scope.
- **R13 en R14 zijn exact.** De documentatie zegt "a maximum of 255 options per Choice" en
  "A Score should have at least two levels; the API accepts up to 10".
- **R9 heeft een geldig precedent.** `BaseModel.generate_image` gooit
  `NotImplementedError(f'generate_image() is not supported by {self.__class__.__name__}')`
  op `basemodel.py:319-321`. `classify` kopieert die vorm letterlijk.
- **Tev1-4B serveert de standaard niet.** Het requirementsdocument noemt het onder "modellen
  die dezelfde vorm serveren", maar Together AI biedt het aan als "chat completions with 2 to
  24 options". Die 24 past op geen van beide grenzen uit de spec. Het model hoort dus bij de
  categorie maar is niet via `classify()` te bereiken. Zie de matrix bij Task 04.
- **De retries zijn niet optioneel.** De documentatie schrijft ze voor: "retry the request
  with exponential backoff instead of retrying immediately", voor 429 en 529. Een classifier
  wordt per definitie in volume aangeroepen, dus dit pad wordt echt geraakt.

## Beantwoorde open vragen

- **OQ1 opgelost.** `POST https://openrouter.ai/api/alpha/decisions` en de slug
  `typesafe/jev-1.13` staan zo in de tutorial, met body `{model, state, questions}` en
  response `{id, model, answers, usage}`. De antwoordvormen voor `choice` en `noul` zijn
  letterlijk gelijk aan AE1 en AE3.
- **OQ2 half opgelost.** Het native pad `https://api.typesafe.ai/v1/systemone` met
  `Authorization: Bearer` staat in de API-reference, dus de tak wordt niet op gevoel
  geschreven. Zonder key blijft hij ongeverifieerd tegen een echte server. De smoke test
  slaat over bij een ontbrekende `TYPESAFE_API_KEY`, zoals de bestaande providertests doen.
- **OQ3 besloten.** De interne vraagnaam voor de enkele-vraagvorm is `answer`, als constante
  `SINGLE` in `systemone.py`. Niet zichtbaar in de API.

## Wat nog niet geverifieerd is

Twee dingen die pas een echte call kunnen beslechten. Beide staan als expliciete stap in
Verificatie.

1. **`state` als platte string op OpenRouter.** TypeSafe documenteert "A plain string for
   text, or structured data", en AE1 geeft een string mee. De OpenRouter-tutorial laat
   uitsluitend een object zien. OpenRouter proxyt naar Jev, dus een string hoort te werken,
   maar als het endpoint een object eist valt AE1 om. Eerste call met een string, en als die
   422 geeft: `state` wordt dan in de OpenRouter-tak in `{'text': state}` gewikkeld, en dat
   wordt een aparte regel in dit plan.
2. **De 529-tak.** Geen enkele provider laat zich op verzoek overbelasten. De mapping wordt
   met een gemockte response getest, niet tegen een echte 529.

## Bestanden

| Bestand | Nieuw | Wat erin komt |
|---|---|---|
| `justai/models/systemone.py` | ja | Wire-formaat, validatie, foutmapping, retries, `SystemOneMixin`, `SystemOneModel` |
| `justai/models/basemodel.py` | nee | `classify()` die `NotImplementedError` gooit |
| `justai/model/model.py` | nee | `Model.classify()` met caching, timing en tellers |
| `justai/models/modelfactory.py` | nee | Routing voor `systemone/*` en `jev*` |
| `justai/models/openrouter_models.py` | nee | `classify()` via het decisions-endpoint |
| `tests/test_classify.py` | ja | Wire-formaat, validatie, foutmapping, zonder netwerk |
| `tests/test_classify_routing.py` | ja | Factory, providerpaden en retries, met gemockte httpx |
| `tests/test_classify_model.py` | ja | `Model.classify`, caching, tellers, `NotImplementedError` |
| `examples/system_one.py` | ja | Voorbeeld met de modellenmatrix in de comments |
| `README.md` | nee | Prefix-tabel plus een feature-sectie |
| `CLAUDE.md`, `AGENTS.md` | nee | Prefix-lijst in de Model Factory-sectie |
| `docs/architecture.md` | nee | Component, routingrijen, cross-cutting regel |

## Execution table

| # | Task | Touches | Depends on |
|---|------|---------|------------|
| 01 | Wire-formaat: requestbuilder, antwoorduitpakker, validatie, foutmapping | `justai/models/systemone.py`, `tests/test_classify.py` | - |
| 02 | Providers: `SystemOneMixin` met retries, `SystemOneModel`, factory-routing, OpenRouter-override | `justai/models/systemone.py`, `justai/models/modelfactory.py`, `justai/models/openrouter_models.py`, `tests/test_classify_routing.py` | 01 |
| 03 | Publieke API: `Model.classify` plus de `BaseModel`-raiser | `justai/model/model.py`, `justai/models/basemodel.py`, `tests/test_classify_model.py` | 01 |
| 04 | Voorbeeldscript | `examples/system_one.py` | 02, 03 |
| 05 | Documentatie | `README.md`, `CLAUDE.md`, `AGENTS.md`, `docs/architecture.md` | 02, 03 |

02 en 03 raken geen gemeenschappelijk bestand en kunnen samen lopen zodra 01 staat. 02 breidt
`systemone.py` uit dat 01 aanmaakt, vandaar de afhankelijkheid. 04 en 05 kunnen samen lopen.

## Task 01: wire-formaat

### Tests eerst, `tests/test_classify.py`

Geen netwerk, geen keys. Alles tegen de pure functies uit `systemone.py`.

| Test | Bewijst |
|---|---|
| `test_dict_options_give_choice` | dict leidt tot `{'type': 'choice', 'criteria': {...}}` (R2) |
| `test_list_options_give_score` | lijst leidt tot `{'type': 'score', 'criteria': [...]}` (R2) |
| `test_no_options_give_noul` | `None` leidt tot `{'type': 'noul'}` zonder `criteria` (R2) |
| `test_instructions_land_in_question` | `instructions=` komt in de vraag, en ontbreekt als hij niet is meegegeven |
| `test_state_accepts_str_dict_list` | drie vormen gaan door de builder (R3) |
| `test_state_rejects_other_types` | een int op `state` is een `AssertionError` (R3) |
| `test_options_rejects_other_types` | een str op `options` is een `TypeError` met het gevonden type in de melding |
| `test_choice_at_255_passes` | 255 opties mag |
| `test_choice_over_255_raises` | 256 opties is een `AssertionError` (R13) |
| `test_score_at_2_and_10_pass` | de randen zijn inclusief |
| `test_score_at_1_and_11_raise` | 1 en 11 niveaus zijn een `AssertionError` (R14) |
| `test_options_and_questions_are_exclusive` | beide meegeven is een `AssertionError` (R4) |
| `test_questions_pass_through_unchanged` | de rauwe dict gaat ongewijzigd de body in (R4) |
| `test_payload_carries_wire_model_name` | `model` staat in de body |
| `test_unpack_single_flattens` | `{'answers': {'answer': X}}` wordt `X` |
| `test_unpack_batch_keeps_map` | bij `questions=` komt de hele `answers`-map terug (R4) |
| `test_unpack_keeps_unknown_fields` | een verzonnen extra veld in het antwoord blijft staan (R5) |
| `test_usage_of_reads_tokens` | `input_tokens` en `output_tokens` komen eruit |
| `test_usage_of_ignores_cost` | `cost` in de response verandert niets aan het resultaat |
| `test_usage_of_survives_missing_usage` | een response zonder `usage` geeft `(0, 0)` |
| `test_map_http_error_per_status` | 401 en 403 naar `AuthorizationException`, 400 en 422 naar `BadRequestException`, 429 naar `RatelimitException`, 529 naar `ModelOverloadException`, 500 naar `ConnectionException`, rest naar `GeneralException` (R12) |
| `test_map_http_error_529_is_not_connection` | 529 valt niet in de generieke 5xx-tak |
| `test_map_http_error_timeout_before_transport` | `httpx.ReadTimeout` geeft `TimeoutException`, niet `ConnectionException` |
| `test_map_http_error_json_decode` | onleesbare json geeft `GeneralException` |

Commit deze tests rood voordat `systemone.py` bestaat. Ze falen dan op de import, wat het
bewijs is dat ze het nieuwe bestand echt aanspreken.

### Daarna implementeren

`justai/models/systemone.py`, alleen functies, nog geen klasse:

- Constanten `MAX_CHOICE_OPTIONS = 255`, `MIN_SCORE_LEVELS = 2`, `MAX_SCORE_LEVELS = 10`,
  elk met de geciteerde documentatieregel als comment. `SINGLE = 'answer'`.
- `build_question(options, instructions) -> dict`. Leidt het type af uit de vorm van
  `options`, assert de grenzen, hangt `instructions` erin als die er is. Een `options` die
  geen dict, lijst of `None` is geeft een `TypeError` met het gevonden type erin.
- `build_payload(wire_model_name, state, options=None, *, instructions=None, questions=None) -> dict`.
  Assert dat `options` en `questions` niet samen gegeven zijn, assert het type van `state`,
  bouwt `{'model': ..., 'state': ..., 'questions': ...}`. Zonder `questions` wordt de enkele
  vraag onder `SINGLE` gezet.
- `unpack(body, single) -> dict`. Bij `single` de ene waarde onder `SINGLE`, anders de hele
  `answers`-map. Filtert niets en dopt niets om (R5).
- `usage_of(body) -> tuple[int, int]`. Leest alleen `input_tokens` en `output_tokens`.
- `map_http_error(e) -> Exception`. Volgorde telt en is geverifieerd: `httpx.TimeoutException`
  erft van `TransportError`, dus die check komt eerst. `httpx.HTTPStatusError` erft niet van
  `TransportError` maar wel van `HTTPError`, dus `HTTPError` is de juiste vangnetklasse voor
  de call. Zelfde comment-vorm als `map_openai_error` in `openai_completions.py:89-93`.
  529 wordt vóór de generieke `>= 500`-tak getest, anders verdwijnt de overload in
  `ConnectionException`.

## Task 02: providers

### Tests eerst, `tests/test_classify_routing.py`

httpx wordt gemockt met `monkeypatch`, er gaat geen request de deur uit. De
`isolate_cache_and_warnings`-fixture uit `tests/test_effort.py:26` wordt gekopieerd. De
retry-tests monkeypatchen `time.sleep` naar een no-op, anders duurt de suite 14 seconden.

| Test | Bewijst |
|---|---|
| `test_factory_routes_systemone_prefix` | `systemone/kev-3b` geeft een `SystemOneModel` |
| `test_factory_routes_jev_prefix` | `jev-1.13.0` geeft een `SystemOneModel` |
| `test_factory_keeps_openrouter_first` | `openrouter/typesafe/jev-1.13` geeft nog steeds een `OpenRouterModel` |
| `test_systemone_requires_base_url` | `systemone/x` zonder `base_url` is een `AssertionError` bij constructie |
| `test_systemone_works_without_api_key` | een lokale server zonder key construeert gewoon |
| `test_systemone_posts_to_base_url` | de URL is `<base_url>/v1/systemone` (R8) |
| `test_base_url_trailing_slash_is_handled` | `http://localhost:8000/` geeft geen dubbele slash |
| `test_jev_posts_to_typesafe` | de URL is `https://api.typesafe.ai/v1/systemone` (R7) |
| `test_jev_sends_bare_model_name` | `systemone/kev-3b` stuurt `kev-3b` in het `model`-veld |
| `test_openrouter_posts_to_decisions` | de URL is `https://openrouter.ai/api/alpha/decisions` (R6) |
| `test_openrouter_sends_bearer_key` | de header bevat de `OPENROUTER_API_KEY` |
| `test_base_url_and_key_stay_out_of_body` | `base_url` en `SYSTEMONE_API_KEY` staan niet in de gepostte json |
| `test_classify_records_usage` | `record_usage` is geraakt met de tokens uit de response (R11) |
| `test_classify_records_usage_before_unpack` | een response met `usage` maar zonder `answers` laat de tellers gevuld achter |
| `test_retries_on_429_then_succeeds` | twee keer 429 en dan 200 geeft het antwoord, met drie calls |
| `test_retries_on_529` | 529 loopt door hetzelfde pad |
| `test_retries_give_up_as_ratelimit` | blijvend 429 eindigt als `RatelimitException` (R12) |
| `test_retries_give_up_as_overload` | blijvend 529 eindigt als `ModelOverloadException` |
| `test_no_retry_on_422` | een 422 doet precies één call |
| `test_max_retries_zero_does_one_call` | `Model(..., max_retries=0)` retryt niet |
| `test_backoff_is_exponential` | de gemockte sleep krijgt 2 en dan 4 |
| `test_systemone_stubs_prompt` | `prompt` op een `SystemOneModel` is een `NotImplementedError` |
| `test_close_closes_http_client` | `Model.close()` sluit de httpx-client van de mixin |
| `test_typesafe_smoke` | echte call, `skipif` zonder `TYPESAFE_API_KEY` (OQ2) |
| `test_openrouter_smoke_string_state` | echte call met een platte `state`-string tegen `openrouter/typesafe/jev-1.13`, `skipif` zonder `OPENROUTER_API_KEY`. Dit is de test die de openstaande vraag over `state` beslecht |

### Daarna implementeren

In `systemone.py` een `SystemOneMixin` met de HTTP-call:

- `classify(state, options=None, *, instructions=None, questions=None) -> tuple[dict, int, int]`.
- Drie hooks die de subclass vult: `classify_url()`, `classify_headers()` en
  `wire_model_name()`.
- **Connection reuse.** De mixin houdt een lazy `httpx.Client` in
  `self._systemone_client`, niet een losse `httpx.post` per call zoals `ReveModel` doet. Een
  TLS-handshake kost 50 tot 100 ms en deze modellen antwoorden in 70 tot 500 ms, dus zonder
  keep-alive gaat een groot deel van de winst die deze modelcategorie oplevert weer weg.
  De mixin overschrijft `close()`: eerst de eigen client, dan `super().close()`. Dat is nodig
  omdat `BaseModel.close()` (`basemodel.py:250-258`) alleen naar `self.client` kijkt, en dat
  attribuut is bij `OpenRouterModel` al de OpenAI-client.
- **Timeout.** `classify` gebruikt niet de `DEFAULT_TIMEOUT` van 120 seconden. Een call die
  in een halve seconde klaar hoort te zijn en na twee minuten nog hangt, is dood. Default 30
  seconden, met `client_timeout` (`basemodel.py:54`) voor de connect-, write- en poolgrenzen,
  en een `timeout=`-parameter die het nog steeds overschrijft.
- **Retries.** 429 en 529 worden opnieuw geprobeerd met `time.sleep(2 ** attempt)`, hetzelfde
  patroon als de Agent in `agent/agent.py:306-317`. Het aantal komt uit `max_retries` in
  `model_params`, met 2 als default zodat het gedrag gelijk is aan wat de provider-SDK's doen
  (`DEFAULT_MAX_RETRIES` in `basemodel.py:26`). Andere statuscodes gaan direct door naar
  `map_http_error`, want een 422 wordt niet beter van wachten.
- Vangt `httpx.HTTPError` en `json.JSONDecodeError` en gooit `map_http_error(e) from e`.
  Niet breder vangen: een `except Exception` zou ook programmeerfouten in de builder
  opslokken.
- Roept `record_usage()` aan voordat `unpack` loopt, want `unpack` kan struikelen over een
  ontbrekende sleutel en die tokens zijn dan al betaald. Dat is dezelfde regel die de
  docstring van `record_usage` in `basemodel.py:238-240` stelt.

`SystemOneModel(SystemOneMixin, BaseModel)` in hetzelfde bestand:

- Twee vormen. `systemone/<naam>` eist een `base_url`-parameter en haalt de key op met
  `get_api_key` onder `SYSTEMONE_API_KEY`, maar mag zonder key door, want de zelf-gehoste
  servers uit de matrix bij Task 04 vragen er meestal niet om. `jev*` gebruikt
  `https://api.typesafe.ai` en eist `TYPESAFE_API_KEY`.
- `base_url` wordt met `rstrip('/')` aan `/v1/systemone` geplakt.
- `wire_model_name()` levert de naam zonder `systemone/`-prefix.
- `supports_return_json`, `supports_image_input` en `supports_tool_use` op `False`. Het model
  produceert geen tekst, dus die drie zijn er niet.
- `prompt`, `chat`, `prompt_async`, `chat_async` en `token_count` gooien
  `NotImplementedError`, met precies het patroon van `ReveModel` in `reve_models.py:83-96`.
- `base_url` moet in `_NON_API_PARAMS`, anders lekt het naar de requestbody. De key wordt al
  door `get_api_key` uit `params` gepopt (`basemodel.py:335`).

`ModelFactory.create` krijgt twee takken, na `openrouter/` zodat
`openrouter/typesafe/jev-1.13` niet op `jev` wordt afgevangen, en voor de slotsom `else`:

```python
elif model_name.startswith('systemone/') or model_name.startswith('jev'):
    from justai.models.systemone import SystemOneModel
    return SystemOneModel(model_name, params=kwargs)
```

`OpenRouterModel` krijgt `SystemOneMixin` als tweede basis, plus `self.api_key = api_key`
(die wordt nu in de OpenAI-client begraven en is daarna niet meer op te vragen). De drie hooks
wijzen naar het decisions-endpoint en naar `self.model_name`, die op dat punt al gestript is
tot `typesafe/jev-1.13`.

Een OpenRouter-slug die geen System One-model is gaat naar hetzelfde endpoint en krijgt daar
een 422, die als `BadRequestException` terugkomt. Dat is bewust: justai kan niet weten welke
van de duizenden OpenRouter-slugs deze vorm serveren, en een ingebakken lijst verjaart binnen
een maand.

## Task 03: publieke API

### Tests eerst, `tests/test_classify_model.py`

| Test | Bewijst |
|---|---|
| `test_basemodel_classify_raises_with_class_name` | de melding bevat de klassenaam (R9) |
| `test_claude_classify_raises` | `Model('claude-sonnet-5').classify(...)` gooit zonder een request te doen (AE6) |
| `test_classify_returns_dict` | de gemockte providerdict komt er onveranderd uit (R1, R5) |
| `test_classify_fills_token_counts` | `input_token_count` en `output_token_count` staan goed (R11) |
| `test_classify_sets_response_time` | `last_response_time` is groter dan nul (R11) |
| `test_classify_caches_by_default` | tweede identieke call raakt de provider niet (R10) |
| `test_classify_cache_survives_dict_result` | de dict komt na een cache-hit als dict terug, niet als str |
| `test_different_options_miss_the_cache` | dezelfde `state` met andere opties gaat opnieuw naar de provider (R10) |
| `test_different_instructions_miss_the_cache` | idem voor `instructions` (R10) |
| `test_different_questions_miss_the_cache` | idem voor een andere `questions`-dict (R10) |
| `test_classify_cached_false_skips_cache` | met `cached=False` gaat elke call naar de provider |
| `test_classify_cache_hit_zeroes_counters` | een hit zet de tellers op nul, want er is niets betaald |
| `test_classify_keeps_tokens_on_failure` | een provider die na `record_usage` faalt laat de tellers gevuld achter |

`test_different_options_miss_the_cache` is de test die het makkelijkst per ongeluk groen
staat. Schrijf hem eerst tegen een cache-key zonder de vraagspecificatie, kijk of hij faalt,
en voeg de specificatie dan toe.

### Daarna implementeren

`BaseModel.classify`, direct onder `generate_image` in `basemodel.py`:

```python
def classify(self, *args, **kwargs):
    """Overwrite in subclasses that DO support classification."""
    raise NotImplementedError(f'classify() is not supported by {self.__class__.__name__}')
```

`Model.classify` in `model.py`, gemodelleerd op `Model.prompt`:

- Signatuur `classify(self, state, options=None, *, instructions=None, questions=None, cached=True)`.
- Klokt zelf de tijd en zet `last_response_time` aan het eind.
- Cache-key: modelnaam, `model_params`, de letterlijke string `'classify'`, `state`, `options`,
  `instructions`, `questions`. Die string houdt classify-entries uit de buurt van
  prompt-entries, die een even lange argumentenlijst hashen.
- Resultaat gaat als JSON de cache in en komt er met `json.loads` weer uit.
- De providercall loopt door `self._call(...)` uit `model.py:153`, zodat de tellers ook bij een
  exception blijven staan.
- Geen capability-vlag op `Model`-niveau. De `NotImplementedError` komt uit `BaseModel`, wat
  R9 vraagt en een vlag minder betekent.

## Task 04: voorbeeld

`examples/system_one.py`, in de vorm van de bestaande voorbeelden: functies per geval, een
`__main__`-blok dat naar de projectroot chdirt en `load_dotenv(override=True)` doet, zoals
`examples/basic.py:17-20`.

Vier functies, een per vraagtype plus de batch: `choice_example`, `score_example`,
`noul_example`, `batch_example`. Ze printen het antwoord, `model.last_token_count()` en
`model.last_response_time`, zodat de snelheidsclaim van deze modelcategorie meteen zichtbaar
is. Draait op `openrouter/typesafe/jev-1.13`, de enige ingang die nu een key heeft.

Een vijfde functie `unsupported_example` roept `classify` op een Claude-model aan en vangt de
`NotImplementedError`, zodat het voorbeeld ook laat zien wat er niet kan.

Bovenaan een commentblok met de modellenmatrix, in drie groepen. De indeling is geverifieerd,
niet overgenomen: de eerste groep serveert de standaard echt, de tweede hoort bij de categorie
maar spreekt een eigen protocol, en de derde is onduidelijk. Zonder die scheiding zou het
voorbeeld modellen aanraden die op `classify()` een 404 geven.

```
# Werkt via classify(), serveert POST /v1/systemone:
#   Jev          TypeSafe, hosted     Model('jev-latest')                     TYPESAFE_API_KEY
#                via OpenRouter       Model('openrouter/typesafe/jev-1.13')   OPENROUTER_API_KEY
#   Kev          Jared Palmer         Model('systemone/kev-3b', base_url=...)   Apache 2.0
#   Laya         Convai              via laya-serve                            Apache 2.0
#   Decider      Mapika                                                        Apache 2.0
#   Von          wfzyx                                                         Apache 2.0
#   Rizzo Flow   Rizzo AI Academy                                              Apache 2.0
#   OpenThai-SystemOne  iApp          mirrort POST /v1/systemone                Apache 2.0
#
# Zelfde modelcategorie, eigen protocol, dus NIET via classify():
#   Tev1-4B      Together AI          chat completions met 2 tot 24 opties
#   GLiNER2.5-Decide  Fastino         eigen classify_text API
#   Bespoke Nimble    Bespoke Labs    eigen schemaformaat
#   SemIf        TheoLeeCJ            leest uit bevroren LLM's
#
# Protocol niet vastgesteld op 26-09-2026, eerst het pad nakijken:
#   AnyJev       Nokia Applied Research    NanoJev   TianyuCodings
```

Het voorbeeld draait niet automatisch mee in de tests, net als de andere voorbeelden.

## Task 05: documentatie

`README.md`:

- Twee rijen in de prefix-tabel op regel 53: `jev*` naar TypeSafe en `systemone/*` naar
  "Self-hosted System One".
- Een sectie `### Classification (System One models)` onder Features, na "JSON and structured
  output". Eén alinea over wat deze modellen zijn en waarom ze anders zijn (geen tekst, wel
  gekalibreerde kansen, tientallen tot honderden milliseconden), dan de drie voorbeelden uit
  AE1 tot AE3, de batchvorm uit AE4, en één regel dat andere modellen
  `NotImplementedError` geven.

`CLAUDE.md` en `AGENTS.md`: de twee nieuwe prefixen in de Model Factory-lijst. Beide
bestanden hebben dezelfde lijst en beide missen nu `minimax*`, dat sinds
`modelfactory.py:49` bestaat. Die rij gaat er in dezelfde beweging bij.

`docs/architecture.md`:

- `systemone.py` bij de componenten in `justai/models/`.
- Twee rijen in de routingtabel, plus de ontbrekende `minimax*`-rij.
- Een cross-cutting regel over classificatie: één wire-formaat, drie ingangen, eigen
  retry-lus met keep-alive omdat de latency het punt van deze modellen is, en de reden dat er
  geen emulatie op gewone LLM's is.
- Twee correcties die tijdens de verificatie bovenkwamen: regel 57 noemt de fixture
  `isolate_cache`, maar hij heet `isolate_cache_and_warnings` (`tests/test_effort.py:26`), en
  de uitbreidingsstap moet ook `AGENTS.md` noemen, niet alleen `README.md` en `CLAUDE.md`.

## Verificatie

1. `venv/bin/pytest tests/test_classify.py tests/test_classify_routing.py tests/test_classify_model.py`
   is groen, zonder dat er een key in `.env` staat. De twee smoke tests slaan dan over en dat
   moet in de output te zien zijn als `skipped`, niet als `passed`.
2. Elke nieuwe test heeft een rode commit voor zijn implementatie. Dat is het bewijs dat de
   test de implementatie echt aanspreekt.
3. `venv/bin/pytest tests/` laat geen bestaande test kapot. De wijziging in
   `openrouter_models.py` is de enige die aan bestaand gedrag raakt, dus `tests/test_effort.py`
   en `tests/test_cache.py` moeten meelopen.
4. Met een `OPENROUTER_API_KEY` in `.env`: `venv/bin/python examples/system_one.py` geeft vier
   antwoorden en een `NotImplementedError` die netjes gevangen wordt. Dit is de enige stap die
   bewijst dat het echte endpoint doet wat de documentatie belooft, en daarmee de afronding van
   OQ1.
5. Diezelfde run beslecht de openstaande vraag over `state`: `choice_example` geeft een platte
   string mee. Komt daar een 422 op, dan wordt de OpenRouter-tak aangepast en komt er een test
   bij die de wrapping vastlegt.
6. `venv/bin/python -c "from justai import Model; Model('claude-sonnet-5').classify('x', {'a': 'b'})"`
   geeft `NotImplementedError: classify() is not supported by AnthropicModel`, letterlijk de
   melding uit AE6.
7. Eén call timen op de echte OpenRouter-ingang. Blijft `last_response_time` boven een seconde
   op de tweede en derde call, dan doet de keep-alive niet wat hij moet doen en is er iets mis
   met de client-levensduur.

`uv` staat niet in het PATH van deze shell, maar wel op `~/.local/bin/uv`. De commando's
hierboven gebruiken daarom `venv/bin/`. Met `uv` op het PATH is `uv run pytest ...`
gelijkwaardig.

## Buiten scope

| Wat | Trigger om het terug te halen |
|---|---|
| Afbeeldingen als input | Zodra een System One-model beeld aankondigt. De documentatie zegt nu "Text only. No image, audio, or video input." |
| `criteria` op `noul` | Zodra iemand de `"true"`- en `"false"`-beschrijvingen nodig heeft. Vraagt een extra parameter, want R2 leidt `noul` juist af uit een ontbrekende `options`. |
| Emulatie op gewone LLM's | Zodra er vraag is naar één codepad over beide modelsoorten en het verlies aan kalibratie acceptabel is. |
| `classify_many` en `classify_async` | Zodra een concreet gebruik de 1200 requests per minuut nadert of in een async app draait. |
| `usage.cost` ontsluiten | Zodra justai kosten over alle providers bijhoudt. Het veld komt binnen bij OpenRouter en wordt nu weggegooid. |
| Streaming | Nooit. Er zijn geen output-tokens om te streamen. |
| Modellen met een eigen protocol | Tev1-4B, GLiNER2.5-Decide, Bespoke Nimble en SemIf vragen elk een eigen adapter. Pas als er vraag naar één is. |
| Classifier in de `Agent`-loop | Eigen plan. |
| `TYPESAFE_API_KEY`-tak tegen een echte server | Zodra er toegang is. Tot dan draait hij op de documentatie en een overgeslagen test. |
