# Doel

JustAI is een Python-package die een uniforme interface biedt voor het werken met Large Language Models van verschillende providers via één enkele `Model`-klasse.

## Wat het oplost

Zonder JustAI moet je per provider (OpenAI, Anthropic, Google, X AI, DeepSeek, Perplexity, Reve, OpenRouter, lokale GGUF) een aparte SDK leren, met verschillende auth, message-formats en response-structuren. Dat is repetitief en foutgevoelig.

Met JustAI kies je een model op naam (`gpt-5-mini`, `claude-sonnet-4-5`, `gemini-2.5-flash`, ...) en werkt de rest hetzelfde: `Model(name).chat(prompt)`.

## Kernprincipes

- **Simpel boven flexibel.** Eén interface, model-naam bepaalt de provider. Geen configuratie-hell.
- **Feature-parity waar mogelijk.** Chat, streaming (`chat_async`), JSON/structured output (`return_json`, `response_format`), multimodaal (PIL, URL, raw), tool calling, prompt caching.
- **Uitbreidbaar via plugin-patroon.** Nieuwe provider = nieuwe klasse in `justai/models/` die `BaseModel` implementeert plus prefix-match in `ModelFactory`.
- **Caching ingebouwd.** `cached=True` slaat prompt+response lokaal op voor snelheid en kostenbesparing.

## Doelgroep

Python-ontwikkelaars die snel LLM-features in hun app willen zetten zonder aan één provider vast te zitten. Ondersteunt ook lokale GGUF-modellen voor offline/private inference.

## Niet-doelen

- Geen agent-framework (LangChain-achtige abstractielagen). Wel een lichte `Agent`-klasse voor tool-loops, maar geen graph-based orchestration.
- Geen eigen model-hosting.
- Geen vendor lock-in: elke provider is optioneel, model wisselen is één regel code.
