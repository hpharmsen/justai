# Plan: Support voor Kimi (Moonshot) modellen

**Datum:** 2026-07-19
**Type:** feature
**Status:** draft (deepened 2026-07-19)

## Deepen-samenvatting

Drie parallelle research-forks (API-status, repo-audit, tokenizer/errors) hebben het plan aangescherpt. Belangrijkste correcties:

1. **Test-filename fout in v1:** conventie is `test_kimi.py`, niet `kimi_test.py`. Ook `isolate_cache` fixture kopiëren.
2. **Thinking mode aanname klopte niet.** Kimi levert reasoning NIET via `delta.reasoning_content` in de async stream (zoals DeepSeek). Wel via `extra_body={'thinking': {...}}` en in het non-streaming response. Concrete follow-up nodig.
3. **429 kan "insufficient funds" betekenen**, niet echt rate-limit. Bekende valkuil bij Moonshot.
4. **Temperature clamp**: Kimi weigert temperature > 1. K2.6/K2.5 accepteren zelfs alleen 0.6 (non-thinking) of 1.0 (thinking). Documenteren.
5. **`chat.completions.parse()` guard werkt al**: base_url check op `openai.com` vangt het correct af. Niets doen.

Domein-migratie: docs staan op `platform.kimi.ai` (301 vanaf `platform.moonshot.ai`), maar API-endpoint blijft `api.moonshot.ai/v1`. Env var blijft `MOONSHOT_API_KEY`.

## Doel

Kimi / Moonshot LLMs toevoegen als provider aan JustAI, zodat gebruikers `Model('kimi-k2.6')` (of vergelijkbaar) kunnen aanroepen zonder verdere configuratie. De Moonshot API is OpenAI-compatible, dus we hergebruiken `OpenAICompletionsModel` net zoals `DeepSeekModel` en `OpenRouterModel` dat doen.

## Context

**API-details** (bron: platform.kimi.ai/docs):
- Endpoint: `https://api.moonshot.ai/v1/chat/completions`
- Auth: Bearer token via `MOONSHOT_API_KEY`
- Wire-format: OpenAI-compatible (`openai` Python SDK werkt direct met `base_url`)
- Features: streaming, tool use (identiek aan OpenAI spec), JSON mode (`json_object` en `json_schema`), multimodaal (image_url + video_url), thinking mode op K3/K2.6/K2.7-code
- Niet-standaard OpenAI features (buiten scope v1): partial mode (prefill), preserved thinking

**Beschikbare modellen** (juli 2026, bevestigd via platform.kimi.ai/docs/api/chat):
- Nieuwe generatie: `kimi-k3`, `kimi-k2.7-code`, `kimi-k2.7-code-highspeed`, `kimi-k2.6`, `kimi-k2.5`
- Legacy: `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`, `moonshot-v1-auto`
- Vision: `moonshot-v1-{8k,32k,128k}-vision-preview` (de nieuwe `kimi-k*` modellen hebben eigen multimodale support)

## Aanpak

Kopieer het DeepSeek-patroon (`justai/models/deepseek_models.py`): één klasse die `OpenAICompletionsModel` erft en alleen `__init__` overschrijft voor auth + base URL. Voeg twee prefix-routes toe in `ModelFactory` (`kimi*` en `moonshot*`), beide naar `KimiModel`. Prefix-collision check: geen bestaande route begint met `k` of `m`, dus veilig.

**Waarom deze aanpak:** simpelste oplossing die werkt. Geen nieuwe abstracties, geen speciale behandeling. Tool use en vision (`image_url`) zijn 100% OpenAI-compatible, en `chat.completions.parse` wordt al door bestaande `openai.com` base_url guard in `openai_completions.py:202` afgevangen met NotImplementedError. Alle andere features werken out of the box.

**Wat we NIET doen in v1 (follow-up tickets):**
- **Thinking mode via `extra_body`.** Kimi K3 forceert reasoning altijd aan, K2.7-code ook, K2.6 configureerbaar via `extra_body={'thinking': {'type': 'enabled', 'keep': 'all'}}`. Reasoning komt terug in `response.choices[0].message.reasoning_content` (non-async) NIET in `delta.reasoning_content`. Voor v1 negeren we reasoning; async streaming pikt het niet automatisch op zoals eerder aangenomen.
- **Partial mode / prefill.** Assistant message prefilling via `{"role": "assistant", "content": "...", "partial": true}` (vergelijkbaar met Anthropic prefill). Nuttig voor JSON forcing en rollenspel-continuatie. Aparte helper nodig.
- **Eigen Kimi tokenizer.** `AutoTokenizer.from_pretrained('moonshotai/Kimi-K2-Instruct', trust_remote_code=True)` voor accurate token counts. Zie risico's.
- **Prompt caching integratie** (Moonshot heeft eigen mechanisme).
- **Effort-mapping** (zoals OpenRouter). Alleen als er vraag naar komt.

## Tests (TDD-first)

**Bestand: `tests/test_kimi.py`** (nieuw; naam volgt de `test_*.py` conventie uit `test_effort.py`, `test_agent.py`, `test_cache.py`, `test_google_json.py`)

Kopieer de `isolate_cache` autouse fixture uit `tests/test_effort.py:22-29` om cross-test cache pollution te voorkomen.

Test 1: model factory routeert correct
```python
def test_factory_routes_kimi_prefix():
    from justai.models.modelfactory import ModelFactory
    from justai.models.kimi_models import KimiModel
    model = ModelFactory.create('kimi-k2.6', MOONSHOT_API_KEY='test')
    assert isinstance(model, KimiModel)

def test_factory_routes_moonshot_prefix():
    model = ModelFactory.create('moonshot-v1-8k', MOONSHOT_API_KEY='test')
    assert isinstance(model, KimiModel)
```

Test 2: KimiModel init zet correcte base_url
```python
def test_kimi_client_base_url():
    from justai.models.kimi_models import KimiModel
    model = KimiModel('kimi-k2.6', params={'MOONSHOT_API_KEY': 'test'})
    assert 'api.moonshot.ai' in str(model.client.base_url)
```

Test 3 (integration, vereist echte API key): capabilities
- Voeg `kimi-k2.6` toe aan `ALL_MODELS` in `tests/capabilities_test.py`
- Draai: `python tests/capabilities_test.py -m kimi-k2.6`
- Verwacht: async, json, tooluse werken. Vision alleen op vision-varianten.

**Verificatie:**
```bash
pytest tests/test_kimi.py
python tests/capabilities_test.py -m kimi-k2.6 -t async json tooluse
```

## Implementatie

### Stap 1: `justai/models/kimi_models.py` (nieuw)

```python
import os

from dotenv import dotenv_values
from openai import OpenAI

from justai.models.basemodel import BaseModel, DEFAULT_TIMEOUT
from justai.models.openai_completions import OpenAICompletionsModel
from justai.tools.display import color_print, ERROR_COLOR


class KimiModel(OpenAICompletionsModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f"You are {model_name}, a large language model trained by Moonshot AI."
        BaseModel.__init__(self, model_name, params, system_message)

        keyname = 'MOONSHOT_API_KEY'
        api_key = params.get(keyname) or os.getenv(keyname) or dotenv_values().get(keyname)
        if not api_key:
            color_print(
                'No Moonshot API key found. Create one at https://platform.moonshot.ai/ and '
                f'set it in the .env file like {keyname}=here_comes_your_key.',
                color=ERROR_COLOR,
            )
        timeout = params.get('timeout', DEFAULT_TIMEOUT)
        self.client = OpenAI(api_key=api_key, base_url='https://api.moonshot.ai/v1', timeout=timeout)

        self.messages = [{'role': 'system', 'content': self.system_message}]

        self.supports_function_calling = True
        self.supports_image_input = True  # kimi-k2.x en vision-varianten; API rejects op non-vision
        self.max_output_tokens = 8192

        # Kimi weigert temperature > 1 (OpenAI staat [0, 2] toe). Clamp naar veilige waarde.
        if self.api_params.get('temperature', 0) > 1:
            self.api_params['temperature'] = 1.0
```

### Stap 2: `justai/models/modelfactory.py` uitbreiden

Voeg toe voor de `else`-branch:

```python
        elif model_name.startswith('kimi') or model_name.startswith('moonshot'):
            from justai.models.kimi_models import KimiModel
            return KimiModel(model_name, params=kwargs)
```

### Stap 3: Documentatie

Concrete regel-referenties uit repo-audit:

- **README.md regel 4** (feature-list): `... OpenRouter and local GGUF models.` → `... OpenRouter, Kimi (Moonshot) and local GGUF models.`
- **README.md regel 14-19** (API-key providers-lijst): nieuwe bullet `Moonshot: [platform.moonshot.ai](https://platform.moonshot.ai/)`
- **README.md regel 22-29** (`.env` blok): `MOONSHOT_API_KEY=your-moonshot-api-key`
- **README.md regel 49-59** (prefix-tabel): nieuwe rij `| \`kimi*\`, \`moonshot*\` | Moonshot |`
- **CLAUDE.md regel 25-33** (Model Factory prefix-lijst): `- \`kimi*\`, \`moonshot*\` → Moonshot`
- **pyproject.toml regel 12** (keywords, optioneel): `"Kimi", "Moonshot"` toevoegen

### Stap 4: Capabilities test uitbreiden

In `tests/capabilities_test.py`: voeg `'kimi-k2.6'` toe aan `ALL_MODELS`.

## Kritische bestanden

- `justai/models/modelfactory.py` — routing
- `justai/models/kimi_models.py` — nieuwe provider (nieuw)
- `justai/models/openai_completions.py` — parent class (readonly, alleen begrijpen)
- `tests/test_kimi.py` — unit tests (nieuw; `test_*.py` conventie)
- `tests/capabilities_test.py` — integration test (edit)
- `README.md` + `CLAUDE.md` — docs

## Risico's / open vragen

1. **Model namen verschuiven snel.** Kimi bracht recent K3 uit; over paar maanden zijn er weer nieuwe. Prefix-match op `kimi*` vangt dit op, geen hardcoded lijst nodig.

2. **Vision op non-vision modellen** geeft API-error. Acceptabel: gebruiker krijgt duidelijke foutmelding van Moonshot. Alternatief: `supports_image_input=False` en gebruiker moet override. Kies voor default `True`; simpelste voor de meeste use cases (kimi-k2.x heeft native multimodaal).

3. **Thinking mode klopt niet zoals eerder aangenomen.** `OpenAICompletionsModel.chat_async` leest wel `delta.reasoning_content`, maar Moonshot zet reasoning NIET in het delta-veld. Het komt terug in `response.choices[0].message.reasoning_content` (non-streaming). De niet-async `prompt()` path (`openai_completions.py:130-132`) pakt alleen `message.content` en gooit reasoning weg. Voor v1 accepteren we dit. Follow-up: reasoning capturen zoals `AnthropicModel` doet.

4. **Tiktoken tokencount** kent Kimi niet, valt terug op `cl100k_base`. Foutmarge ~10-20% voor Engels, tot 2x fout voor Chinees (Kimi tokenizer heeft ~160K vocab vs 100K). Voor grove indicatie prima. Follow-up: `AutoTokenizer.from_pretrained('moonshotai/Kimi-K2-Instruct')` voor accurate counts.

5. **429 kan "insufficient funds" betekenen.** Moonshot retourneert 429 zowel voor echte rate limits als voor uitgeput tegoed. Justai vangt dit als `RatelimitException` af. Overweeg de body-error message door te geven in de exception voor duidelijkheid. Follow-up.

6. **Temperature-quirk.** Kimi weigert `temperature > 1`. K2.6/K2.5 accepteren zelfs alleen 0.6 (non-thinking) of 1.0 (thinking). De clamp in `__init__` vangt de bovengrens; strengere per-model regels documenteren of pas na eerste gebruikersklacht toevoegen.

7. **HTTP-error mapping.** OpenAI SDK vertaalt statuscodes correct: 400 → BadRequestError, 401 → AuthenticationError, 429 → RateLimitError, 408 → APITimeoutError, 500 → InternalServerError. Alle bestaande justai catch-blocks werken zonder aanpassing.

## Definition of done

- [ ] `pytest tests/test_kimi.py` slaagt (factory + init, unit tests met MagicMock)
- [ ] `python tests/capabilities_test.py -m kimi-k2.6 -t async json tooluse` slaagt (met echte `MOONSHOT_API_KEY` in `.env`)
- [ ] README.md en CLAUDE.md geüpdatet op de gespecificeerde regels
- [ ] Handmatig getest: `Model('kimi-k2.6').prompt('Hallo')` geeft antwoord
- [ ] Handmatig getest: `Model('moonshot-v1-8k').prompt('Hallo')` geeft antwoord (legacy prefix)

## Follow-up tickets (uit deepen-onderzoek)

- [ ] Thinking mode via `extra_body={'thinking': {'type': 'enabled'}}` + reasoning capture in `prompt()` path (analoog aan AnthropicModel)
- [ ] Partial mode helper voor assistant prefill
- [ ] Eigen Moonshot tokenizer voor accurate token counts (vooral voor Chinees)
- [ ] 429-body-message doorgeven bij insufficient-funds vs echte rate limit
- [ ] Per-model temperature-validatie voor K2.5/K2.6 (0.6 of 1.0)
