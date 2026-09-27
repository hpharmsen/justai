---
title: SDK-parse() valideert in de provider en vervangt je schema
date: 2026-09-27
tags: [structured-output, pydantic, anthropic, openai, sdk, validation-retries]
files:
  - justai/models/anthropic_models.py
  - justai/models/openai_responses.py
  - justai/model/model.py
  - tests/test_structured_passthrough.py
---

# SDK-parse() valideert in de provider en vervangt je schema

## Aanleiding

Voor `validation_retries` (plan `docs/plans/2026-09-27-001-feat-validation-retries-plan.md`,
gemerged als "Merge branch 'feat-validation-retries'") moest `Model` een ongeldig structured
antwoord terug kunnen sturen naar het model, met de fouten erbij. Dat ging niet zolang de
Anthropic- en OpenAI-provider de `parse()`-methode van hun SDK gebruikten.

Een falende `field_validator` gaf daar een `GeneralException`. Het rauwe antwoord was weg,
de tokens van die call waren niet geteld en het antwoord stond niet in de chathistorie.
Precies de drie dingen die een repair-loop nodig heeft.

## De kern

**`parse()` valideert al binnen de SDK-call.** Anthropic's `messages.parse()` hangt een
`post_parser` aan de request die `TypeAdapter(output_format).validate_json(text)` doet
(anthropic 0.96.0, `anthropic/lib/_parse/_response.py`). OpenAI's `responses.parse()` doet
hetzelfde via `model_parse_json` (openai 2.32.0, `openai/lib/_parsing/_responses.py`).

Een validator die faalt, raist dus in `client.messages.parse(...)`, nog voor justai iets
met het antwoord kan doen. In de provider gebeurt dan dit:

- de exceptie landt in het generieke `except Exception` en wordt een `GeneralException`
  (`_map_anthropic_error`, `map_openai_error`);
- `record_usage` staat ná de call en wordt nooit bereikt, terwijl de tokens wel betaald zijn;
- de rauwe tekst en het assistant-bericht verdwijnen met de exceptie.

**En Anthropic's `parse()` gooit je eigen schema stil weg.** De oude code bouwde een schema
in `output_config.format` en riep daarna `messages.parse(output_format=cls, ...)` aan. Maar
`parse()` bouwt zelf `transform_schema(TypeAdapter(cls).json_schema())` en overschrijft
daarmee `format` in je `output_config`. Wie dat handgebouwde schema aanpaste, veranderde
niets aan de request. Geen fout, geen waarschuwing. Het stond gewoon in de code en zag er
plausibel uit.

## Wat niet werkte

- **Aannemen dat het handgebouwde schema verstuurd werd.** Pas de SDK-bron lezen liet zien
  dat `parse()` het vervangt.
- **Een test die de private module blokkeert via `sys.meta_path`.** Bedoeld om te bewijzen
  dat de provider nog importeert als een SDK-upgrade zijn private helper hernoemt. Die test
  is vacuüm: de SDK importeert `anthropic.lib._parse._transform` en
  `openai.lib._parsing._responses` zelf al (de traceback liep via `anthropic/__init__.py`).
  Blokkeren breekt dan de SDK, niet justai, en de test faalde op oude én nieuwe code. Wat wel werkt: eerst de SDK laden, dan alleen het
  attribuut verwijderen, dan de provider importeren
  (`test_providers_import_without_private_sdk_helpers`).

## Wat je doet

Laat de provider `create()` aanroepen en de rauwe JSON-tekst teruggeven. Valideer op één
plek, in `Model` (`_to_pydantic`, en `_validated` voor de repair-loop).

Het schema blijft gelijk aan wat `parse()` zou sturen, door dezelfde helpers te gebruiken:

```python
# Anthropic, in _completion_with_structured_output
from anthropic.lib._parse._transform import transform_schema
schema = transform_schema(response_format.model_json_schema())

# OpenAI, in prompt()
from openai.lib._parsing._responses import type_to_text_format_param
text = {'format': type_to_text_format_param(response_format)}
```

Beide zijn private SDK-modules. Importeer ze daarom pas binnen de Pydantic-tak: hernoemt een
upgrade ze, dan breekt alleen structured output met een Pydantic-class en niet de hele
provider.

Tijdens de bouw zijn de request bodies van `parse()` en `create()` met een
`httpx.MockTransport` naast elkaar gelegd: byte-identiek. Die vergelijking staat niet als
test in de repo; wat er wel vastligt, is dat het verstuurde schema gelijk is aan wat
`parse()` bouwt (`tests/test_structured_passthrough.py`).

Let op de bijwerking: een falende validator geeft nu bij alle providers Pydantic's
`ValidationError` in plaats van `GeneralException`. Dat staat in de README en gaat mee in de
minor release 5.8 (bij het schrijven nog niet gepubliceerd; `pyproject.toml` stond op 5.7.1).

## Toepasbaarheid

- **Gebruik geen SDK-`parse()` in een provider die usage, de rauwe tekst of historie moet
  bijhouden.** Alles wat je na de call wilde doen, verdwijnt met de exceptie.
- Gebruik je een SDK-helper die zelf request-parameters samenstelt, lees dan de bron na of hij
  jouw parameters niet stil overschrijft. Dat geeft geen fout, alleen een andere request dan
  je denkt.
- De tests in `tests/test_structured_passthrough.py` pinnen dat `parse` niet wordt
  aangeroepen, welk schema er meegaat en dat de private helpers bestaan. Faalt
  `test_private_sdk_helpers_exist` na een SDK-upgrade, zoek dan de nieuwe plek van de helper
  in de `parse()`-bron van die SDK.
