---
title: Prompt caching faalt stil, dus bouw de meting mee
date: 2026-09-08
tags: [prompt-caching, anthropic, agent, kosten, api]
files:
  - justai/models/anthropic_models.py
  - justai/models/basemodel.py
---

# Prompt caching faalt stil, dus bouw de meting mee

## Aanleiding

Anthropic mailde dat onze cache hit rate laag was, met een geschatte besparing van
tot 56% op het directe API-verkeer. Justai bleek precies één breakpoint te zetten,
achter een handmatig gezette `cached_prompt`, en alleen in het niet-streaming pad.
De agent-loop cachete niets.

## De kern

**Een gemiste cache geeft geen foutmelding.** Je krijgt een geldig antwoord, alleen
een hogere rekening. Er is geen exception, geen waarschuwing en niets aan de response
dat afwijkt. Dat maakt drie dingen waar die bij normale bugs niet gelden:

1. **Zonder teller weet je niets.** Daarom staan `cache_read_input_tokens` en
   `cache_creation_input_tokens` nu op `BaseModel` in plaats van achter een
   `hasattr`-dans, en vullen OpenAI en Gemini ze ook.
2. **Een unit test bewijst alleen waar het breakpoint staat, niet dat de provider
   het accepteert.** Onder de minimumdrempel (1024 tokens voor Sonnet 4.6, 4096 voor
   Opus 4.6 en Haiku 4.5) cachet de API stil niets. Vandaar de opt-in live test.
3. **Muteren van de aanroeper zijn messages is de gevaarlijkste bug in dit gebied.**
   De Agent hergebruikt zijn `messages`-lijst tussen iteraties. Eén `cache_control`
   die blijft hangen verandert de bytes van de prefix, waarna elke volgende request
   mist. Alles blijft werken, alleen duurder. `apply_cache_control()` is daarom
   copy-on-write, met een test die het bewijst.

## Het kostenmodel bepaalt waar het breakpoint hoort

Een cache write kost 1,25x, een read 0,1x. Bij één message valt er niets te
hergebruiken, dus een breakpoint daar is puur 25% verlies. Vandaar de regel:
**pas vanaf de tweede message**. Dat is geen optimalisatie maar de reden dat
"altijd aan" verdedigbaar is zonder one-shot gebruikers te benadelen.

## Gemeten resultaat

Agent-pad, vier iteraties, prefix van 15,7K tokens:

| Iteratie | Vol tarief | Write | Read |
|---|---|---|---|
| 1 | 25 | 15.710 | 0 |
| 2 | 3 | 37 | 15.710 |
| 3 | 3 | 13 | 15.747 |
| 4 | 3 | 15 | 15.760 |

Omgerekend naar vol-tarief-equivalenten: 63.000 werd 24.500, dus 61% minder. Per
iteratie vanaf de tweede is het circa 90%.

## Valkuil bij het lezen van de cijfers

Bij Anthropic staan de gecachte tokens **naast** `input_tokens`, bij OpenAI en
Gemini zijn ze er een **deelverzameling** van. Wie de twee families optelt in
hetzelfde dashboard telt of dubbel, of mist het grootste deel van de prompt. De
volledige promptomvang bij Anthropic is
`input_token_count + cache_read + cache_creation`.

## Wat we niet gedaan hebben

- Geen breakpoint tussen de laatste twee beurten in. De API kijkt maximaal twintig
  content-blokken terug om een eerdere entry te vinden; een agent-iteratie met veel
  parallelle tool-calls kan daaroverheen gaan en breekt de keten dan stil. Twee van
  de vier breakpoints zijn in gebruik, dus er is ruimte. Pas doen als de meting laat
  zien dat het gebeurt.
- Geen accumulatie van de cachecijfers in `AgentResult`. De `done`-chunk draagt ze
  mee, dus per iteratie is het afleesbaar.
