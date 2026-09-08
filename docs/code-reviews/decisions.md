# Beslissingen

Wat in een code-review-run definitief is afgewogen en niet gaat veranderen. Eén regel per
beslissing, met de reden erbij. Findings op de "Nog te doen"-lijst horen hier niet thuis.

## 2026-07-25

- `identify_image_format_from_base64` delegeert naar `detect_mime_type`, maar mapt BMP
  expliciet terug naar `image/jpeg` — `detect_mime_type` herkent BMP, en geen enkele
  LLM-API accepteert dat formaat. Zonder die terugmapping was de dedup een stille
  gedragswijziging. (2026-07-25)
- `AsyncAnthropic` wordt niet gesloten door `BaseModel.close()` — de SDK biedt daar alleen
  een async `close()` voor, en een synchrone `__exit__` heeft geen draaiende event loop.
  Bewust overgeslagen in plaats van er een `asyncio.run()` omheen te bouwen. Staat als
  comment bij de methode. (2026-07-25)
- De brede `except Exception: pass` in `CacheDB.write` en `CacheDB.clear` blijft staan.
  De cache mag nooit de reden zijn dat een LLM-call faalt; het comment
  ("Whatever, just don't add to the cache but never crash") legt dat al vast. (2026-07-25)
- De vijftien WHY-comments in de contrast-lijst van `docs/code-reviews/latest.md` blijven
  staan. Ze leggen vast waarom `CACHE_NAMESPACE` gebumpt moet worden, waarom `effort`
  geseed wordt, waarom de refusal-check vóór `content[0]` staat, waarom altijd base64 en
  nooit een URL naar OpenAI gaat, en waarom Kimi's temperature geclampt wordt. Dat soort
  context is niet uit de code af te leiden. Een volgende run ruimt ze niet op. (2026-07-25)
- `ReveModel` implementeert `prompt`, `chat`, `prompt_async`, `chat_async` en
  `token_count` als vijf `NotImplementedError`-regels. Dat is geen dode code maar de prijs
  van de abstracte interface in `BaseModel`: een image-only provider moet ze declareren.
  Laten staan. (2026-07-25)
