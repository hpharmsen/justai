---
date: 2026-09-26
topic: classify-system-one
input_source: prompt
input_text: |
  Er is iets nieuws in de AI-wereld, namelijk modellen die geen tokens als output genereren, maar meer classifiers zijn.

  Zoals bijvoorbeeld Jev.

  Zie https://openrouter.ai/docs/guides/community/jev-tutorial

  Ik wil dit toevoegen aan Justai met een nieuwe method: Model.classify

  Justai moet in ieder gevan met Jev werken (voorlopig via OpenRouter omdat ik nog geen toegang heb tot Jev) maar er zijn meer modellen die op basis van dit principe werken.
  Dus ik wil dat Just-I, net als bij Model.prompt, generiek werkt.

  Ga dus eerst op zoek naar vergelijkbare modellen en probeer een grote gemeene dealer te vinden.
  De basis is, de input is tekst of eventueel afbeeldingen erbij, de output is json (omgezet naar een Python dict) met het antwoord en/of de kansverdeling over de mogelijke antwoorden.
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-brainstorm
execution: code
---

# Model.classify: System One-modellen in justai

## Goal Capsule

**Objective.** Een gebruiker van justai kan een tekst laten indelen in een zelf gedefinieerde optieset en krijgt het gekozen antwoord plus de gekalibreerde kansverdeling terug, zonder te weten welke aanbieder erachter zit en zonder een tweede SDK te leren.

**Means.** Eén nieuwe method `Model.classify()` op de bestaande `Model`-klasse, met daarachter een provider-implementatie die de `/v1/systemone`-standaard spreekt.

**Product authority.** De vijf richtinggevende keuzes zijn in de brainstorm van 26 september 2026 gemaakt en staan onder Key Decisions. Wat daar niet staat is aan de planningsfase.

**Open blockers.** Geen. Twee onderdelen zijn niet volledig verifieerbaar en staan onder Outstanding Questions als `Deferred to Planning`.

## Product Contract

### Summary

Justai krijgt `Model.classify()`, een method die tekst voorlegt aan een System One-model en een Python dict teruggeeft met het gekozen antwoord, de confidence en de kansverdeling over de opties. De method werkt op Jev via OpenRouter, op Jev native zodra die toegang er is, en op elk ander model dat `POST /v1/systemone` serveert. Modellen die geen System One-model zijn gooien een duidelijke `NotImplementedError`.

### Problem Frame

Sinds september 2026 bestaat er een nieuwe modelcategorie die TypeSafe "System One models" noemt. Zo'n model genereert geen tekst. Het leest een stuk context, leest de next-token scores over de labels die jij vooraf hebt toegestaan, en geeft daar een gekalibreerde kansverdeling over terug. Dat levert drie dingen op die een gewone LLM niet kan: antwoorden in 70 tot 500 milliseconden in plaats van seconden, ongeveer 0,001 dollar per beslissing met gratis output-tokens, en de garantie dat er geen vierde optie uit de lucht komt vallen omdat het model er maar drie kan kiezen.

Voor het soort werk waar justai nu een LLM voor inzet met `return_json=True` en een prompt vol instructies, denk aan tickets routeren, urgentie inschatten of intent bepalen, is dat een wezenlijk andere prijs-en-snelheidsklasse. Maar het wire-formaat past niet in `chat.completions`: er is geen messages-array, geen assistant-rol en geen content-string. Het is een eigen endpoint met een eigen request- en responsevorm. Zonder aparte method blijft die categorie buiten justai.

De categorie is bovendien in twee weken een veld geworden. Naast Jev serveren minstens tien andere modellen dezelfde endpoint-vorm, waarvan de meeste open weights. Een implementatie die alleen op Jev mikt mist dat veld, terwijl het formaat identiek is.

### Key Decisions

- **Enkele vraag is de primaire vorm, een batch kan via `questions=`.** De standaard is een batch van benoemde vragen, maar het gros van de aanroepen stelt er één. Constrains R1, R2, R4.
- **`classify()` op gewone LLM's gooit `NotImplementedError`.** Een LLM kan wel een label kiezen, maar de kansen die het daarbij noemt zijn verzonnen en zeggen niets over hoe vaak het model het goed heeft. Gekalibreerde kansen zijn precies de waarde van deze modelcategorie, dus een emulatie die er hetzelfde uitziet maar dat niet levert is schadelijker dan een fout. Governs R9.
- **Eén klasse voor de standaard, met instelbare base_url.** Jev native, Kev, Laya, Decider, Von, Tev1 en zelf-gehoste modellen spreken allemaal `POST /v1/systemone`. Het verschil tussen die backends is een base_url en een api-key, niet een formaat. Governs R7, R8.
- **Het vraagtype volgt uit de vorm van het opties-argument.** De standaard onderscheidt de drie types zelf al aan de vorm van `criteria`: een dict bij choice, een lijst bij score, en bij noul geen optieset. Een aparte `type=`-parameter zou die informatie verdubbelen. Governs R2.
- **Geen bulk-voorziening in v1.** Wie duizenden items classificeert schrijft vier regels `ThreadPoolExecutor`. Een eigen `classify_many` moet daarnaast rate limiting, retries en gedeeltelijk falen regelen, en dat is pas de moeite als het gebruik er is. Zie Scope Boundaries voor de trigger.

### Requirements

**API-oppervlak**

- **R1.** `Model.classify(state, options=None, *, instructions=None, questions=None, cached=True)` geeft een Python dict terug.
- **R2.** Zonder `questions` volgt het vraagtype uit `options`: een dict geeft `choice`, een lijst geeft `score`, en geen `options` geeft `noul`.
- **R3.** `state` accepteert `str`, `dict` of `list[str]`, de drie vormen die de spec toestaat.
- **R4.** `questions=` neemt een dict van benoemde vraagspecificaties in het rauwe formaat van de standaard en geeft een dict terug met per vraagnaam het antwoord. `options` en `questions` sluiten elkaar uit; beide meegeven is een fout.
- **R5.** De velden die de provider in een antwoord teruggeeft worden ongewijzigd doorgegeven, zonder omdopen of filteren, zodat modellen die extra velden leveren niet stukgaan op justai.

**Providerdekking**

- **R6.** `openrouter/typesafe/jev-*` werkt via OpenRouters eigen pad `POST /api/alpha/decisions`, met de bestaande `OPENROUTER_API_KEY`.
- **R7.** Een modelnaam die met `jev` begint gaat naar `api.typesafe.ai/v1/systemone` met een `TYPESAFE_API_KEY`.
- **R8.** `systemone/<naam>` gaat naar `<base_url>/v1/systemone`, waarbij `base_url` als parameter wordt meegegeven.
- **R9.** `BaseModel.classify()` gooit `NotImplementedError` met de klassenaam erin, zoals `generate_image()` dat al doet, zodat elk ander model duidelijk faalt in plaats van iets te verzinnen.

**Aansluiting op wat justai al doet**

- **R10.** `cached=True` is de default, gelijk aan `Model.prompt`. De cache-key bevat de vraagspecificatie, anders geeft een tweede vraag over dezelfde state het antwoord van de eerste.
- **R11.** `input_token_count`, `output_token_count` en `last_response_time` worden gevuld, via het bestaande `record_usage()`.
- **R12.** Fouten komen terug als de exceptions die justai al heeft, dus `ConnectionException` en `RatelimitException`, niet als een rauwe provider-exception.

**Validatie vooraf**

- **R13.** Een assert op maximaal 255 opties bij `choice`, de grens uit de spec.
- **R14.** Een assert op 2 tot en met 10 niveaus bij `score`, de grens uit de spec.

### Key Flows

**Eén vraag stellen.** De gebruiker maakt een `Model` met een System One-modelnaam en roept `classify()` aan met de state en de opties. Justai leidt het vraagtype af uit de vorm van de opties (R2), verpakt het als een enkele benoemde vraag in het standaardformaat, stuurt het naar het pad dat bij deze provider hoort (R6, R7 of R8), en pakt het ene antwoord weer uit tot een platte dict.

**Meerdere vragen over dezelfde state.** De gebruiker geeft `questions=` mee (R4). Justai stuurt de dict door zoals hij is en geeft `answers` terug zoals het terugkomt. Het model beantwoordt de vragen server-side parallel in één keer lezen, dus tien vragen kosten niet tien requests.

**Een model dat het niet kan.** De gebruiker roept `classify()` aan op een Claude- of GPT-model en krijgt meteen `NotImplementedError` met de klassenaam (R9). Geen request, geen kosten.

### Acceptance Examples

**AE1, choice.** Covers R1, R2, R6.

```python
m = Model('openrouter/typesafe/jev-1.13')
m.classify('Mijn payouts falen al 3 dagen',
           {'payments': 'geld en uitbetalingen',
            'frontend': 'UI en weergave',
            'account': 'inloggen'},
           instructions='Welk team pakt dit op?')
# {'type': 'choice', 'choice': 'payments', 'confidence': 0.67,
#  'probabilities': {'payments': 0.78, 'frontend': 0.22, 'account': 0.0}}
```

**AE2, score.** Covers R2, R14.

```python
m.classify(ticket, ['kan wachten', 'normaal', 'nu'], instructions='Hoe urgent?')
# {'type': 'score', 'score': 1.99, 'confidence': 0.99,
#  'probabilities': {'0': 0.0, '1': 0.0, '2': 1.0},
#  'legend': {'0': 'kan wachten', '1': 'normaal', '2': 'nu'}}
```

**AE3, noul.** Covers R2.

```python
m.classify(ticket, instructions='Is dit een softwarefout?')
# {'type': 'noul', 'noul': 0.96}
```

**AE4, batch.** Covers R4.

```python
m.classify(ticket, questions={
    'team': {'type': 'choice', 'instructions': 'Welk team?',
             'criteria': {'payments': '...', 'frontend': '...'}},
    'urgentie': {'type': 'score', 'instructions': 'Hoe urgent?',
                 'criteria': ['laag', 'midden', 'hoog']}})
# {'team': {...}, 'urgentie': {...}}
```

**AE5, zelf-gehost.** Covers R8.

```python
m = Model('systemone/kev-3b', base_url='http://localhost:8000')
m.classify(text, {'ja': '...', 'nee': '...'}, instructions='...')
# zelfde dict-vorm als AE1
```

**AE6, niet ondersteund.** Covers R9.

```python
Model('claude-sonnet-5').classify(text, {'a': '...', 'b': '...'})
# NotImplementedError: classify() is not supported by AnthropicModel
```

### Scope Boundaries

**Binnen scope.** De drie vraagtypes van de standaard, de drie provider-ingangen uit R6 tot R8, en de aansluiting op caching, tellers en exceptions uit R10 tot R12.

**Buiten scope, met trigger.**

- *Afbeeldingen als input.* De opdracht noemde ze, maar het kan niet: TypeSafe documenteert expliciet "Text only. String, JSON object, or array of text values. No image, audio, or video input." De blogpost zegt "not on images (yet…)". Terug op de agenda zodra een System One-model beeld aankondigt.
- *Emulatie op gewone LLM's.* Zie de tweede Key Decision. Terug op de agenda als er vraag is naar één codepad dat over beide modelsoorten werkt en het verlies aan kalibratie acceptabel is.
- *`classify_many` en `classify_async`.* Terug op de agenda zodra er een concreet gebruik is dat de rate limit van 1200 requests per minuut nadert of dat in een async app draait.
- *Kosten ontsluiten.* De response bevat `usage.cost` in dollars, maar justai heeft nergens een kostenbegrip. Terug op de agenda als justai kosten over alle providers gaat bijhouden, niet voor deze ene.
- *Streaming.* Er zijn geen output-tokens om te streamen.
- *De `Agent`-klasse.* Een classifier als router binnen een tool-loop is een eigen stuk werk.

### Outstanding Questions

- **OQ1, `Deferred to Planning`.** `POST /api/alpha/decisions` bij OpenRouter is een alpha-endpoint. Het pad en de slug `typesafe/jev-1.13` komen uit de tutorial van september 2026 en kunnen wijzigen. De planningsfase verifieert het pad tegen een echte call voordat R6 als klaar geldt.
- **OQ2, `Deferred to Planning`.** R7 kan niet getest worden zolang er geen TypeSafe-toegang is. De tak wordt geschreven op basis van de documentatie en blijft ongeverifieerd tot er een key is. De planningsfase bepaalt of die tak achter een test komt die overslaat bij een ontbrekende key, zoals de bestaande provider-tests doen.
- **OQ3, `Deferred to Planning`.** Welke vraagnaam justai intern gebruikt om een enkele vraag in het `questions`-formaat te verpakken en weer uit te pakken (R1, R4). Puur intern, niet zichtbaar in de API.

### Landschap

Referentiemateriaal voor de planningsfase, peildatum 26 september 2026.

**De standaard.** `POST /v1/systemone`, body is `state` plus `questions`, response is `answers` plus `usage`. Drie vraagtypes: `choice` geeft de winnaar plus een kans per optie, `score` geeft een kansgewogen positie op een geordende schaal, en `noul` geeft één kans tussen 0 en 1. De optie- en niveaugrenzen staan op R13 en R14. Verdere limieten bij Jev: 64k tokens per request, waarvan 32k voor `state` plus de langste vraag, 1200 requests per minuut, 0,042 dollar per miljoen input-tokens en gratis output. Engels werkt het best, andere talen zijn minder nauwkeurig.

**Modellen die dezelfde vorm serveren.** Jev (TypeSafe, hosted, `jev-1.13.0` met aliassen `jev-latest` en `jev-preview`), Kev (Jared Palmer, open weights op Qwen3.5 en Qwen3.8), Laya (Convai, open weights op ModernBERT en mmBERT), Decider (Mapika), Von (wfzyx), Tev1-4B (Together AI), OpenThai-SystemOne (iApp), Rizzo Flow (Rizzo AI Academy). Daarnaast met een eigen interface: GLiNER2.5-Decide (Fastino, 340M encoder), Bespoke Nimble, SemIf, AnyJev en NanoJev.

**Bronnen.** [OpenRouter Jev-tutorial](https://openrouter.ai/docs/guides/community/jev-tutorial), [TypeSafe modeldocumentatie](https://docs.typesafe.ai/models), [TypeSafe aankondiging](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [systemonemodels.org](https://systemonemodels.org/guides/what-is-a-system-one-model/), [awesome-system-one](https://github.com/yanng981/awesome-system-one).
