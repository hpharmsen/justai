# Code Review — 2026-07-25 /Users/hp/proj/justai (hele codebase)

## Samenvatting
- Totaal findings: 41 (SAFE: 29, LIKELY: 11, RISKY: 1)
- **Toegepast: 40 van de 41.** Alleen de RISKY-finding (7.2) blijft handwerk.
- **Resultaat: 799 regels weg, 256 erbij, over 25 bestanden.**
- Testbaseline vóór wijzigingen: **53 passed, 1 failed, 1 skipped**
  (`tests/test_effort.py::test_smoke_effort_low_reaches_provider` faalde op een echte
  netwerkcall naar OpenAI — stond al rood, niet veroorzaakt door deze review).
  Na afloop: **54 passed, 1 skipped**.
- Zes gedragsproblemen opgelost, waarvan één pas tijdens het toepassen gevonden (zie
  bijvangst 2.4b).

## Bijvangst

Vijf findings zijn geen reductie maar een echt gedragsprobleem. Ze staan bovenaan omdat
ze anders verdrinken in een lijst van eenenveertig.

- **4.8** — `justai.__version__` is `None`. `_get_version()` zet het resultaat in een
  lokale variabele en vergeet de `return`, dus het happy path levert niets op.
  Geverifieerd: `import justai; justai.__version__` → `None`.
- **5.1** — `model.generate_image(prompt)` zonder `size` crasht op gpt-image modellen.
  `_pick_image_api_size(None)` valt door naar `fit_size(None)` → `TypeError: cannot
  unpack non-iterable NoneType`. De `if not size: return 'auto'`-guard die dit hoort af
  te vangen staat in het onbereikbare deel van de functie (zie 4.3).
- **5.4** — `raise_for_unsupported()` wordt op twee plekken zonder argumenten aangeroepen,
  terwijl beide checks op die argumenten hangen. De guard vuurt dus nooit: `return_json=True`
  op een model dat het niet ondersteunt komt ongehinderd door.
- **5.5** — de prefill-strip voor `NO_PREFILL_MODELS` doet niets. Regel 426 hernoemt de
  lokale `api_messages` naar een nieuwe lijst, maar `api_params['messages']` wijst nog naar
  de oude. De trailing assistant-message gaat alsnog mee naar de API.
- **2.4** — de vriendelijke "No X API key found"-melding is onbereikbaar bij 7 van de 8
  providers. `dotenv_values()[keyname]` gooit een `KeyError` vóór de `if not api_key`-check.
  Alleen Kimi gebruikt `.get()` en haalt de melding wel.
- **2.4b** — *gevonden tijdens het toepassen, niet bij de scan.*
  `OpenAIResponsesModel` haalde de API-sleutel netjes op en gaf hem vervolgens nooit door:
  `self.client = OpenAI(timeout=...)` zonder `api_key=`. Het werkte alleen omdat de
  OpenAI-SDK zelf `OPENAI_API_KEY` uit de omgeving leest. Gevolg:
  `Model('gpt-5', OPENAI_API_KEY='...')` negeerde die sleutel stil, terwijl
  `OpenAICompletionsModel` hem wél honoreert. De ruff-gate legde dit bloot als
  `F841 api_key is assigned but never used` zodra de helper uit 2.4 de toekenning
  zichtbaar maakte. Opgelost door `api_key=api_key` mee te geven.

---

## 1. Zware dependencies (0 findings)

Geen. De dependencies (anthropic, openai, google-genai, httpx, PIL, tiktoken) dragen elk
hun gewicht. Zie 3.1 voor het enige twijfelgeval (`jsonschema`), dat als speculatieve
abstractie is geclassificeerd omdat de library wél gebruikt wordt, alleen te zwaar.

## 2. Duplicate code / DRY (SAFE: 3, LIKELY: 4)

### Finding 2.1 [SAFE] — Python→JSON typemap staat er vijf keer
- Regels bespaard: ~30
- Locaties: `justai/agent/agent.py:64`, `justai/agent/agent.py:379-382`,
  `justai/models/anthropic_models.py:665-672`, `justai/models/openai_completions.py:360-367`,
  `justai/models/openai_responses.py:524-531`
- Huidig (vijf keer, met wisselende naam `type_map` / `type_mapping`):
  ```python
  type_mapping = {int: "integer", str: "string", float: "number",
                  bool: "boolean", list: "array", dict: "object"}
  ```
- Voorstel: één constante in `basemodel.py`, overal importeren:
  ```python
  JSON_TYPE_MAP: dict[type, str] = {
      str: 'string', int: 'integer', float: 'number',
      bool: 'boolean', list: 'array', dict: 'object',
  }
  ```
- Actie: constante toevoegen, vijf lokale kopieën vervangen, `_python_type_to_json()`
  in agent.py vervangen door `JSON_TYPE_MAP.get(t, 'string')`
- Bewijs voor het weghalen van `_python_type_to_json`: `git grep -n "_python_type_to_json"`
  geeft alleen definitie + één aanroep, geen string-literal treffers

### Finding 2.5 [SAFE] — twee implementaties van MIME-detectie via magic bytes
- Regels bespaard: ~18
- Locaties: `justai/tools/images.py:93` (`detect_mime_type`, neemt bytes),
  `justai/models/basemodel.py:205` (`identify_image_format_from_base64`, neemt base64)
- Beide herkennen PNG/GIF/JPEG/WebP met dezelfde magic bytes en dezelfde JPEG-fallback.
  `identify_image_format_from_base64` mist BMP, `detect_mime_type` mist de GIF87a-variant
  niet maar wel de commentaarregel over 12 bytes.
- Voorstel: `identify_image_format_from_base64` delegeert:
  ```python
  def identify_image_format_from_base64(encoded_data: str) -> str:
      """Identify image format from base64 data. Returns MIME type supported by LLM APIs."""
      from justai.tools.images import detect_mime_type
      return detect_mime_type(base64.b64decode(encoded_data)[:12])
  ```
- Actie: body vervangen. Beide namen en signatures blijven staan, dus de
  dispatch-checklist is niet van toepassing.

### Finding 2.6 [SAFE] — drie functies met hetzelfde `match image_type`-blok
- Regels bespaard: ~12
- Locaties: `justai/tools/images.py:76` (`to_base64_image`), `:108` (`to_base64_data_uri`),
  `:129` (`to_pil_image`)
- Alle drie doen dezelfde drietrapsconversie (URL fetchen / bytes doorgeven / PIL opslaan).
- Voorstel: één private `_to_bytes(image) -> bytes`, de drie publieke functies bouwen
  daarop. `to_pil_image` houdt zijn shortcut voor `pil_image` (geen roundtrip door JPEG).
- Actie: helper toevoegen, drie match-blokken inkorten

### Finding 2.2 [LIKELY] — Anthropic-foutmapping staat er drie keer
- Regels bespaard: ~45
- Locaties: `justai/models/anthropic_models.py:322-349`, `:438-462`, `:564-580`
- Huidig: hetzelfde zevental `except`-takken (APIConnection / Auth / InternalServer /
  RateLimit / BadRequest / APIStatus 503+529 / Exception) → dezelfde justai-excepties.
- Voorstel: één `_map_anthropic_error(e) -> Exception` plus een `@contextmanager`, of een
  enkele `except Exception as e: raise _map_anthropic_error(e)`.
- **Waarom LIKELY**: de drie kopieën loggen verschillend. `completion()` gebruikt kale
  `print(...)`, `_completion_with_structured_output()` gebruikt `logger.error(...)`,
  `stream()` logt helemaal niet. Unificeren verandert wat de gebruiker op stdout ziet, en
  `_completion_with_structured_output` heeft bovendien twee takken die bewust
  doorgooien (`TypeError/AttributeError` en de "does not support output format"-BadRequest)
  die niet mogen sneuvelen.

### Finding 2.3 [LIKELY] — OpenAI-foutmapping staat er vier keer
- Regels bespaard: ~35
- Locaties: `justai/models/openai_completions.py:243-256`, `:424-435`,
  `justai/models/openai_responses.py:151-162`, `:434-445`
- Voorstel: één `_map_openai_error(e)` in `openai_completions.py`, geïmporteerd door
  `openai_responses.py`.
- **Waarom LIKELY**: `openai_completions.py:253` heeft een extra `except NotImplementedError:
  raise` die de andere drie niet hebben. Die tak bestaat omdat `completion()` zelf
  `NotImplementedError` gooit op regel 203/205; verdwijnt hij, dan wordt een bewuste
  "niet ondersteund"-melding stilletjes een `GeneralException`.

### Finding 2.4 [LIKELY] — API-key boilerplate in acht providers
- Regels bespaard: ~40 (en lost bijvangst op)
- Locaties: `anthropic_models.py:95-105`, `openai_responses.py:63-66`,
  `openai_completions.py:85-88`, `google_models.py:61-65`, `deepseek_models.py:20-23`,
  `kimi_models.py:18-24`, `perplexity_models.py:19-22`, `xai_models.py:32-36`,
  `openrouter_models.py:21-25`, `reve_models.py:38-41`
- Huidig, met kleine variaties:
  ```python
  api_key = params.get(keyname) or os.getenv(keyname) or dotenv_values()[keyname]
  if not api_key:
      color_print("No X API key found. Create one at <url> and "
                  f"set it in the .env file like {keyname}=here_comes_your_key.", color=ERROR_COLOR)
  ```
- Voorstel: één helper in `basemodel.py`:
  ```python
  def get_api_key(params: dict, keyname: str, provider: str, url: str) -> str | None:
      key = params.get(keyname) or os.getenv(keyname) or dotenv_values().get(keyname)
      if not key:
          color_print(f'No {provider} API key found. Create one at {url} and '
                      f'set it in the .env file like {keyname}=here_comes_your_key.', color=ERROR_COLOR)
      return key
  ```
- **Waarom LIKELY**: dit is niet puur een dedup maar ook een gedragswijziging.
  `dotenv_values()[keyname]` → `.get(keyname)` verandert een `KeyError`-crash in de
  bedoelde waarschuwing plus een doorloop met `api_key=None`. Dat is vrijwel zeker de
  bedoeling (waarom staat die melding er anders), maar het betekent wel dat een ontbrekende
  sleutel nu pas bij de eerste API-call knalt in plaats van bij constructie.
  Geverifieerd: `dotenv_values()['NONEXISTENT']` → `KeyError`.

### Finding 2.7 [LIKELY] — `token_count` identiek in twee OpenAI-klassen
- Regels bespaard: ~8
- Locaties: `justai/models/openai_completions.py:346-353`, `justai/models/openai_responses.py:598-605`
- Beide: `tiktoken.encoding_for_model` met `cl100k_base`-fallback op `KeyError`.
- Voorstel: naar een module-functie of naar een gedeelde mixin.
- **Waarom LIKELY**: `OpenAIResponsesModel` en `OpenAICompletionsModel` hebben geen gedeelde
  ouder behalve `BaseModel`, en `XAIModel` erft van de eerste terwijl `DeepSeek`/`Kimi`/
  `Perplexity`/`OpenRouter` van de tweede erven. De methode naar `BaseModel` tillen geeft
  tiktoken-gedrag aan Google, Anthropic en Reve, waar het onjuist is. De vorm van de fix
  is dus een ontwerpkeuze: module-functie of mixin.

## 3. Speculatieve abstracties / YAGNI (SAFE: 1, LIKELY: 1)

### Finding 3.1 [LIKELY] — `is_valid_json_schema` is 70 regels voor één assert
- Regels bespaard: ~60
- Locatie: `justai/models/openai_responses.py:772-841`, aangeroepen op `:135`
- Huidig: kiest een validator via `$schema`, draait `check_schema()`, draait daarna
  nogmaals volledige meta-validatie om *alle* fouten te verzamelen, en heeft twee
  interne formatters (`_format_iter_error`, `_format_schema_error`) inclusief
  `oneOf`/`anyOf` contextafhandeling.
- Enige aanroeper:
  ```python
  ok, errors = is_valid_json_schema(response_format)
  assert ok, f"Response format should be a valid JSON Schema or Pydantic model: {errors}"
  ```
- Voorstel:
  ```python
  try:
      Draft202012Validator.check_schema(response_format)
  except exceptions.SchemaError as e:
      raise ValueError(f'response_format is not a valid JSON Schema: {e.message}') from e
  ```
- **Waarom LIKELY**: de foutmelding wordt korter en noemt alleen het eerste probleem in
  plaats van alle. Voor een gebruiker die een schema van honderd regels doorgeeft is dat
  merkbaar slechter. Daarnaast: `assert` verdwijnt bij `python -O`; vervangen door een
  echte `raise` is de betere kant op, maar dat is een gedragswijziging op een pad dat
  geen test raakt.

### Finding 3.2 [SAFE] — `WebFetchTool.__init__` doet niets
- Regels bespaard: 2
- Locatie: `justai/agent/tools/web_fetch.py:66-67`
- Huidig:
  ```python
  def __init__(self):
      pass
  ```
- Actie: weghalen. De impliciete `object.__init__` doet hetzelfde.

## 4. Dode code (SAFE: 11, LIKELY: 1)

### Finding 4.2 [SAFE] — `transform_messages` in openai_responses.py heeft nul aanroepers
- Regels bespaard: 45
- Locatie: `justai/models/openai_responses.py:552-596`
- Bewijs: `git grep -n "transform_messages"` geeft alleen de twee definities (deze en 4.1),
  geen enkele aanroep, en geen string-literal treffer. De methode bouwt nog het oude
  Chat-Completions berichtformaat (`msg["tool_calls"]`, `msg["tool_call_id"]`) dat de
  Responses API niet eens accepteert; het interne TODO op regel 560 bevestigt dat tool use
  hier weg is verhuisd.
- Actie: methode verwijderen. Daarmee wordt de import van `ToolUseMessage` op regel 32
  ook ongebruikt (zie 4.10).

### Finding 4.1 [SAFE] — `transform_messages` in openai_completions.py heeft nul aanroepers
- Regels bespaard: 45
- Locatie: `justai/models/openai_completions.py:295-339`
- Bewijs: zelfde grep als 4.2. Zelfde TODO op regel 303.
- Actie: methode verwijderen

### Finding 4.3 [SAFE] — onbereikbare staart in `_pick_image_api_size`
- Regels bespaard: 13
- Locatie: `justai/models/openai_responses.py:674-686`
- Huidig: regel 671-672 doet `w, h = fit_size(size); return f'{w}x{h}'`. Daarna volgt een
  losse docstring plus een tweede, volledige implementatie (`if not size: return 'auto'`,
  de 1024x1024/1536x1024-tabel) die nooit draait.
- Actie: regels 674-686 verwijderen
- Let op: de `if not size`-guard in dit dode blok is precies wat bijvangst 5.1 zou
  oplossen. Voer 5.1 uit voordat of nadat dit weggaat, maar laat het niet liggen.

### Finding 4.6 [SAFE] — `Agent._write_tasks` heeft nul aanroepers
- Regels bespaard: 11
- Locatie: `justai/agent/agent.py:242-252`
- Bewijs: `git grep -n "_write_tasks"` geeft alleen de definitie. `run()` leest het
  tasks-bestand wel (`_read_tasks`, regel 277) maar schrijft het nooit terug; regel 362-363
  yieldt alleen een `task_update`-event zonder iets naar schijf te zetten.
- Actie: methode verwijderen, plus de imports `os` en `tempfile` die daarmee ongebruikt
  raken (regel 5-6)

### Finding 4.7 [SAFE] — `print_message` en `DEBUG_COLOR1` hebben nul gebruikers
- Regels bespaard: 7
- Locatie: `justai/tools/display.py:9`, `:17-22`
- Bewijs: `git grep -n "print_message"` en `git grep -n "DEBUG_COLOR"` geven voor
  `print_message` alleen de definitie, en voor `DEBUG_COLOR1` alleen de toekenning.
  `DEBUG_COLOR2` wordt wél gebruikt (`openai_completions.py:145`) en blijft staan.
  `USER_COLOR`/`ASSISTANT_COLOR`/`SYSTEM_COLOR` worden alleen binnen `print_message`
  gebruikt en gaan mee.
- Actie: `print_message`, `DEBUG_COLOR1`, `ASSISTANT_COLOR`, `SYSTEM_COLOR`, `USER_COLOR`
  verwijderen. `color_print` en `ERROR_COLOR` blijven.

### Finding 4.10 [SAFE] — 17 ongebruikte imports buiten `__init__.py`
- Regels bespaard: ~10
- Locaties (ruff `F401`):
  `agent/agent.py:3` (`json`), `:14` (`StreamChunk`);
  `model/message.py:3` (`PIL.Image`), `:5` (`is_image_url`);
  `model/model.py:2` (`json`), `:5` (`Optional`, `Union`, `List`), `:11` (`Message`);
  `models/anthropic_models.py:37` (`APITimeoutError`), `:67` (`Message`);
  `models/basemodel.py:19` (`Message`);
  `models/deepseek_models.py:2` (`Any`), `:7` (`Message`);
  `models/google_models.py:34` (`Message`);
  `models/openai_responses.py:37` (`UNIVERSAL_EFFORT_LEVELS`);
  `models/perplexity_models.py:6` (`Message`)
- Actie: `ruff check --select F401 --fix` op deze bestanden. Geen van deze is een
  re-export (dat zijn alleen de `__init__.py`'s, zie 4.5), dus per de dispatch-checklist
  is dit SAFE zonder verdere toets.

### Finding 4.11 [SAFE] — dubbele imports in openai_responses.py
- Regels bespaard: 2
- Locatie: `justai/models/openai_responses.py:21` en `:23` importeren beide uit `typing`
  (`Any` staat er twee keer in); `:33` en `:35` importeren beide uit
  `justai.models.basemodel`.
- Actie: samenvoegen tot één `from typing import ...` en één basemodel-import

### Finding 4.4 [SAFE] — `if __name__ == '__main__'`-blok in `justai/__init__.py`
- Regels bespaard: 6
- Locatie: `justai/__init__.py:38-43`
- Huidig:
  ```python
  if __name__ == '__main__':
      # Onderstaande om de voorkomen dat import optimizer ze leeg gooit
      a = Model
      g = get_prompt
      ...
  ```
- Dit is een truc tegen IDE-import-optimizers, geen code die ooit draait: een
  package-`__init__.py` krijgt nooit `__name__ == '__main__'` bij normaal gebruik.
- Actie: weghalen, samen met de losse comment op regel 34-36 over `importlib.version`.
  Vervalt tegen 4.5: `__all__` doet hetzelfde werk correct.

### Finding 4.5 [SAFE] — `justai/__init__.py` publiceert 17 namen zonder `__all__`
- Regels bespaard: geen — dit is een finding waar de uitkomst *meer* code is
- Locatie: `justai/__init__.py:5-18`
- Alle 17 imports zijn bedoelde re-exports (`Model`, de negen exception-types, `Agent` en
  vrienden, de drie tools, de drie prompt-functies), maar zonder `__all__` ziet zowel ruff
  als een import-optimizer ze als dood.
- Actie: `__all__` toevoegen met de 17 namen plus `__version__`. Dat maakt het publieke
  oppervlak expliciet, laat ruff zwijgen, en maakt 4.4 overbodig.
- Niet meegeteld in "regels bespaard".

### Finding 4.12 [SAFE] — ongebruikte lokale `ptype` in `extract_images`
- Regels bespaard: 1
- Locatie: `justai/tools/images.py:25` (ruff `F841`)
- `ptype = getattr(part, "type", None)` wordt toegekend en nooit gelezen.
- Actie: regel verwijderen

### Finding 4.8 [SAFE] — `_get_version()` vergeet zijn `return` (bijvangst)
- Regels bespaard: 0 — dit is een bugfix
- Locatie: `justai/__init__.py:21-32`
- Huidig:
  ```python
  def _get_version():
      try:
          __version__ = version(__name__)     # <- lokale variabele, ruff F841
      except PackageNotFoundError:
          ...
  ```
  Het happy path zet een lokale naam en valt eruit met `None`. Geverifieerd:
  `import justai; justai.__version__` → `None`.
- Voorstel:
  ```python
  def _get_version() -> str:
      try:
          return version(__name__)
      except PackageNotFoundError:
          with open(Path(__file__).parent / 'pyproject.toml') as f:
              for line in f:
                  if line.startswith('version ='):
                      return line.split('"')[1]
          raise RuntimeError('Unable to find version')
  ```
- Actie: `return` toevoegen. Let op dat de `raise RuntimeError` nu buiten de `with` moet
  staan zodat hij ook echt bereikbaar is.

### Finding 4.9 [LIKELY] — `Message.from_dict` en `Message.to_dict` hebben nul aanroepers
- Regels bespaard: ~16
- Locatie: `justai/model/message.py:28-33`, `:45-50`
- Bewijs: `git grep -n "to_dict\|from_dict"` geeft alleen de definities. Ook geen
  string-literal treffer.
- **Waarom LIKELY**: `Message` staat niet in `justai/__init__.py` en niet in de README,
  maar is wel gewoon importeerbaar als `justai.model.message.Message` en wordt door
  meerdere providers geïmporteerd. Een downstream gebruiker die `Message` gebruikt om
  een conversatie te serialiseren, gebruikt precies deze twee methodes. Bovendien doet
  `from_dict` een `setattr(message, key, value)` met een niet-literale key: dat is de ene
  dynamic-dispatch-plek in de codebase, en hij wijst naar de `Message`-namespace zelf.

## 5. Defensive checks voor onmogelijke scenarios (SAFE: 1, LIKELY: 4)

### Finding 5.2 [SAFE] — onbereikbare `case _` in drie image-converters
- Regels bespaard: 6
- Locaties: `justai/tools/images.py:88-89`, `:123-124`, `:138-139`
- Huidig: alle drie doen `image_type = get_image_type(image)` en hebben daarna
  `case _: raise ValueError(f"Unknown image type: {image_type}")`. Maar `get_image_type`
  (regel 65-73) gooit zélf al een `ValueError` voor alles buiten de drie bekende types,
  dus het vierde geval is per constructie onbereikbaar.
- Actie: de drie `case _`-takken verwijderen

### Finding 5.1 [LIKELY] — `generate_image` zonder `size` crasht (bijvangst)
- Regels bespaard: -1 — dit is een regel erbij
- Locatie: `justai/models/openai_responses.py:613`, aangeroepen vanaf `:690`
- `model.generate_image('een kat')` op een gpt-image model geeft
  `TypeError: cannot unpack non-iterable NoneType object`. Geverifieerd door
  `_pick_image_api_size(None)` direct aan te roepen.
- Voorstel: guard bovenaan `_pick_image_api_size` zetten:
  ```python
  if not size:
      return 'auto'
  ```
- **Waarom LIKELY**: `'auto'` is wat de dode implementatie in 4.3 teruggaf, maar of dat
  nog de juiste waarde is voor de huidige Images API is niet uit de code af te leiden.
  De alternatieven zijn een expliciete `ValueError` ("size is verplicht voor gpt-image")
  of een default-formaat. Dat is een keuze, geen feit.

### Finding 5.4 [LIKELY] — `raise_for_unsupported()` vuurt nooit (bijvangst)
- Regels bespaard: 0 of 6, afhankelijk van de richting
- Locatie: `justai/model/model.py:223-227`, aangeroepen op `:140` en `:168`
- Huidig:
  ```python
  def raise_for_unsupported(self, images: ImageInput = None, return_json=False):
      if return_json and not self.model.supports_return_json: raise ...
      if images and not self.model.supports_image_input: raise ...
  ```
  Beide aanroepers doen `self.raise_for_unsupported()` — zonder argumenten. Met de
  defaults `None` en `False` is elke tak dood.
- Twee zinnige richtingen:
  - **Argumenten doorgeven**: `self.raise_for_unsupported(images, return_json)` in
    `prompt()` en `chat()`. De guard gaat dan werken zoals bedoeld. Nul regels bespaard,
    maar `return_json=True` op bijvoorbeeld Perplexity (`supports_return_json = False`)
    gaat dan een nette `NotImplementedError` geven in plaats van een kapotte JSON-parse.
  - **Methode weghalen**: als de providers hun eigen checks al doen (en dat doen ze deels,
    zie `openai_completions.py:184-187`), is dit dubbel werk. 6 regels bespaard.
- **Waarom LIKELY**: richting één zet een guard aan die nu uit staat. Welke calls daardoor
  gaan falen is niet te overzien zonder alle `supports_*`-vlaggen per provider langs te
  gaan, en geen test raakt dit pad.

### Finding 5.5 [LIKELY] — prefill-strip is een no-op (bijvangst)
- Regels bespaard: 0 — bugfix
- Locatie: `justai/models/anthropic_models.py:414-426`
- Huidig:
  ```python
  api_params = {'model': ..., 'messages': api_messages, **self.api_params}
  ...
  if api_messages and api_messages[-1]['role'] == 'assistant' and NO_PREFILL_MODELS.search(self.model_name):
      api_messages = api_messages[:-1]     # <- nieuwe lijst, api_params wijst nog naar de oude
  ```
- Voorstel: `api_params['messages'] = api_messages[:-1]`, of de strip vóór de opbouw van
  `api_params` doen.
- **Waarom LIKELY**: de fix zet een guard aan die vandaag niets doet. Voor
  `claude-opus-4-[6-9]` en `claude-sonnet-4-[6-9]` in chat-modus verandert daarmee de
  payload: de laatste assistant-message verdwijnt. Dat is de bedoeling van
  `NO_PREFILL_MODELS`, maar het is wel een echte gedragswijziging op een pad zonder test.

### Finding 5.3 [LIKELY] — `Model.close()` guard voor een methode die nergens bestaat
- Regels bespaard: 3
- Locatie: `justai/model/model.py:61-64`
- `if hasattr(self.model, 'close'): self.model.close()` — geen enkele provider-klasse
  definieert `close()`. `Model.close()` en daarmee de hele `__exit__` doen dus niets.
- **Waarom LIKELY**: dit is een vooruitziende guard, geen vergissing. De httpx-clients in
  `AnthropicModel` (regel 110-117) zouden juist wél gesloten moeten worden. De juiste
  uitkomst is waarschijnlijk niet "guard weg" maar "`close()` implementeren op de
  providers die een client bezitten". Dat is meer code, niet minder.

## 6. Verbose patterns (SAFE: 5)

### Finding 6.2 [SAFE] — drie geneste `if`-blokken (ruff SIM102)
- Regels bespaard: 3
- Locaties: `justai/models/anthropic_models.py:615-616`, `justai/tools/images.py:38-39`,
  `justai/tools/images.py:153-154`
- Actie: samenvoegen tot één `if` met `and`

### Finding 6.5 [SAFE] — `load_skills` bouwt een lijst met een expliciete loop
- Regels bespaard: 3
- Locatie: `justai/agent/skills.py:14-20`
- Huidig:
  ```python
  if not md_files:
      return ''
  parts = []
  for f in md_files:
      parts.append(f.read_text().strip())
  return '\n\n'.join(parts)
  ```
- Voorstel: `return '\n\n'.join(f.read_text().strip() for f in md_files)` — de
  lege-lijst-guard is dan overbodig, `join` op een lege generator geeft al `''`.

### Finding 6.4 [SAFE] — `try/except/pass` waar `contextlib.suppress` hoort (ruff SIM105)
- Regels bespaard: 2
- Locatie: `justai/tools/cache.py:124-127`
- Actie: `with contextlib.suppress(sqlite3.OperationalError): cur.execute(...)`

### Finding 6.1 [SAFE] — `if`/`else` waar een ternary past (ruff SIM108)
- Regels bespaard: 2
- Locatie: `justai/agent/agent.py:213-216`
- Actie: `result = func(ctx, **tc.arguments) if needs_ctx else func(**tc.arguments)`

### Finding 6.3 [SAFE] — `key in dict.keys()` (ruff SIM118)
- Regels bespaard: 0
- Locatie: `justai/models/basemodel.py:115`
- Actie: `{k for k in params if k not in {'effort'}}`

## 7. Nutteloze wrappers (LIKELY: 1, RISKY: 1)

### Finding 7.1 [LIKELY] — `Model.prompt_async` laat `images` vallen
- Regels bespaard: 0 of 4
- Locatie: `justai/model/model.py:183-186`
- Huidig:
  ```python
  async def prompt_async(self, prompt, *, images: ImageInput = None):
      async for content, reasoning in self.model.prompt_async(prompt=prompt):
          yield content, reasoning
  ```
  De parameter `images` staat in de signature, wordt gedocumenteerd door zijn type hint,
  en wordt vervolgens genegeerd. Vergelijk `chat_async` er direct onder, die `images` wél
  doorgeeft én normaliseert naar een lijst.
- Twee richtingen: `images` doorgeven (zoals `chat_async`), of de parameter schrappen zodat
  de signature niet meer liegt.
- **Waarom LIKELY**: `prompt_async` is publiek en wordt in `examples/async.py` gebruikt.
  De parameter schrappen breekt elke aanroeper die hem meegeeft (die nu stil genegeerd
  wordt); hem doorgeven verandert het gedrag van precies die aanroepers. Welke van de twee
  juist is hangt af van of image-input op het async pad ondersteund hoort te zijn.

### Finding 7.2 [RISKY] — de doorgeef-façade op `Model`
- Regels bespaard: ~20 als alles weg zou gaan
- Locatie: `justai/model/model.py:210-221`, `:229-230`
- `stream`, `format_tool_result`, `format_assistant_message` en `token_count` doen niets
  dan doorgeven aan `self.model`. `Agent` omzeilt ze trouwens en gaat rechtstreeks naar
  `self.model.model.stream(...)` (`agent.py:295`, `:323`, `:338`).
- **Waarom RISKY**: dit is het gedocumenteerde publieke oppervlak van de package. Of
  `Model` een dunne façade blijft of dat `Agent`'s directe `.model.model`-toegang juist de
  fout is, is een ontwerpbeslissing die verder reikt dan deze regels. Geen auto-fix,
  geen interactieve vraag.

## 8. Comment-hygiëne (SAFE: 8)

### Finding 8a.2 [SAFE] — 109 regels uitgecommentarieerde streaming-events
- Regels bespaard: 109
- Locatie: `justai/models/openai_responses.py:283-391`
- Een volledige dump van SSE-events uit de OpenAI-docs, inclusief een enkele regel van
  1300 tekens. Staat in de docs van OpenAI en veroudert hier stilletjes mee.
- Actie: verwijderen, eventueel vervangen door één regel met de docs-URL

### Finding 8a.1 [SAFE] — 57 regels uitgecommentarieerde voorbeeld-response
- Regels bespaard: 57
- Locatie: `justai/models/openai_responses.py:207-263`
- Idem: een JSON-voorbeeld uit de OpenAI-docs, geplakt na het `return`-statement van
  `prompt()`.
- Actie: verwijderen

### Finding 8a.3 [SAFE] — uitgecommentarieerde code in openai_completions.py
- Regels bespaard: 9
- Locaties: `:124-128` (een `if response_format:`-blok met uitleg waarom het níét zo moet),
  `:341-344` (`tool_use_message`)
- Actie: beide verwijderen

### Finding 8a.4 [SAFE] — uitgecommentarieerde `instructor.patch`
- Regels bespaard: 5
- Locatie: `justai/models/openai_responses.py:68-72`
- "Not sure if this works, or is needed, for the Responses API" — een twijfel uit een
  vorige iteratie, geen instructie voor de volgende lezer.
- Actie: verwijderen

### Finding 8a.5 [SAFE] — uitgecommentarieerde `raise` in `prompt()`
- Regels bespaard: 2
- Locatie: `justai/models/openai_responses.py:113-114`
- Actie: verwijderen

### Finding 8a.6 [SAFE] — debug-comments in het Nederlands
- Regels bespaard: 0
- Locatie: `justai/tools/images.py:24` (`# Hier komt ie`), `:39` (`# Hier komt ie ook`)
- Overblijfselen van een debugsessie; ze markeren welke tak destijds geraakt werd.
- Actie: verwijderen

### Finding 8a.7 [SAFE] — losse notitie over `importlib.version`
- Regels bespaard: 3
- Locatie: `justai/__init__.py:34-36`
- Actie: verwijderen (vervalt met 4.4)

### Finding 8b.1 [SAFE] — WHY ontbreekt bij twee `for _ in range(3)`-loops
- Regels erbij: 2
- Locaties: `justai/models/anthropic_models.py:372`, `justai/models/openai_completions.py:216`
- Beide lezen als een retry-loop maar zijn het niet: het is een tool-use-vervolgloop, en de
  drie is het maximum aantal rondes waarin het model functies mag aanroepen voordat we
  stoppen. In `openai_responses.py:126` staat dat wél toegelicht
  (`# Max 3 function calls to prevent infinite loop`); hier niet.
- Actie: dezelfde toelichting toevoegen op beide plekken

### Comments die bewust blijven staan

Zonder deze lijst ruimt een volgende run ze alsnog op. Elk van deze legt een WHY vast die
niet uit de code volgt:

- `tools/cache.py:16-18` — waarom `CACHE_NAMESPACE` gebumpt moet worden bij wijzigingen aan
  `_NON_API_PARAMS`. Het waardevolste comment in de repo: zonder dit levert een upgrade
  stille cache-hits op verkeerde sleutels.
- `tools/cache.py:145`, `:165` — "Whatever, just don't add to the cache but never crash".
  Documenteert dat de brede swallow een bewust ontwerp is.
- `models/basemodel.py:110-111` — waarom `effort` als default geseed wordt: anders routeert
  `Model.__setattr__` de schrijfactie niet naar `model_params`.
- `models/basemodel.py:114` — waarom `_user_supplied` bestaat: auto-raise mag expliciete
  waarden niet overschrijven.
- `models/basemodel.py:207` — "Need 12 bytes for WebP detection".
- `models/anthropic_models.py:48-49` — "Extend this list rather than adding more
  NO_X_MODELS constants". Een ontwerprichtlijn voor de volgende wijziging.
- `models/anthropic_models.py:158` — waarom `max_tokens` omhoog gaat bij hoge effort.
- `models/anthropic_models.py:180-182` — waarom structured outputs alleen bij expliciete
  `response_format`.
- `models/anthropic_models.py:204`, `:464` — waarom de refusal-check vóór `content[0]` moet.
- `models/openai_responses.py:44-45` — waarom de OpenAI-vocabulaire op `xhigh` gecapt is.
- `models/openai_responses.py:513-514` — waarom altijd base64 en nooit een URL
  ("Some servers (like Wikipedia) block OpenAI's download attempts").
- `models/google_models.py:42` — waarom alleen Gemini 3.x `thinking_level` krijgt.
- `models/google_models.py:235` — waarom tekst niet geyield wordt bij function calls.
- `models/kimi_models.py:34` — waarom temperature geclampt wordt op 1.
- `agent/tools/web_fetch.py:18` — "Link-local / cloud metadata" bij `169.254.0.0/16`.

---

## Applied-log (Modus 1, SAFE-set)

Verificatie-gate na de SAFE-set:

| Stap | Uitkomst |
|---|---|
| `ruff check --select F,E9 justai/` | **All checks passed** (eerst rood: zie 2.1 hieronder) |
| Framework-check | n.v.t., geen Django |
| Testsuite | **54 passed, 1 skipped** — baseline was 53 passed, **1 failed**, 1 skipped |
| Import-smoke (alle 25 modules) | **schoon**; `gguf_models` overgeslagen, optionele `llama_cpp` |
| Entrypoint-smoke | n.v.t., library-package zonder console_script |

De test die in de baseline rood stond (`test_smoke_effort_low_reaches_provider`) slaagt nu.
Die faalde op een timeout in een echte netwerkcall naar OpenAI, niet op code; reken hem
niet als winst van deze review.

| Finding | Status | Wat er is gebeurd |
|---|---|---|
| 2.1 | applied | `JSON_TYPE_MAP` in `basemodel.py`; vijf kopieën vervangen, `_python_type_to_json` weg |
| 2.5 | applied | `identify_image_format_from_base64` delegeert naar `detect_mime_type` |
| 2.6 | applied | `_to_bytes()` helper; drie `match`-blokken ingekort |
| 3.2 | applied | lege `WebFetchTool.__init__` weg |
| 4.1 | applied | `transform_messages` uit openai_completions.py |
| 4.2 | applied | `transform_messages` uit openai_responses.py |
| 4.3 | applied | onbereikbare staart van `_pick_image_api_size` weg |
| 4.4 | applied | `if __name__ == '__main__'`-blok weg |
| 4.5 | applied | `__all__` in de drie `__init__.py`'s |
| 4.6 | applied | `Agent._write_tasks` weg, plus `os`/`tempfile` imports |
| 4.7 | applied | `print_message` + vier ongebruikte kleurconstanten weg |
| 4.8 | applied | `_get_version()` geeft nu een `return`; geverifieerd `justai.__version__ == '5.5.4'` |
| 4.10 | applied | 17 ongebruikte imports weg |
| 4.11 | applied | dubbele `typing`- en basemodel-imports samengevoegd |
| 4.12 | applied | ongebruikte `ptype` weg |
| 5.2 | applied | drie onbereikbare `case _`-takken weg |
| 6.1 | applied | SIM108 ternary in `_execute_tool` |
| 6.2 | applied | drie SIM102-nestingen platgeslagen |
| 6.3 | applied | SIM118 `.keys()` weg |
| 6.4 | applied | `contextlib.suppress` in `cache.py` |
| 6.5 | applied | `load_skills` als generator-expressie |
| 8a.1 | applied | 59 regels voorbeeld-JSON weg |
| 8a.2 | applied | 110 regels SSE-events weg |
| 8a.3 | applied | uitgecommentarieerd `response_format`-blok + `tool_use_message` weg |
| 8a.4 | applied | `instructor.patch`-notitie weg |
| 8a.5 | applied | uitgecommentarieerde `raise` weg |
| 8a.6 | applied | "Hier komt ie"-debugcomments weg |
| 8a.7 | applied | `importlib.version`-notitie weg |
| 8b.1 | applied | WHY-comment bij beide `for _ in range(3)`-loops |

**Netto: 486 regels weg, 99 erbij, over 18 bestanden.**

Twee correcties tijdens het toepassen, allebei omdat het label net te optimistisch was:

- **2.1** liet de gate rood achter: `_python_type_to_json` had nog één aanroeper op
  `agent.py:124` die mijn grep in het rapport wel toonde maar die ik bij het schrappen
  oversloeg. Opgelost door daar `JSON_TYPE_MAP.get(v, 'string')` te gebruiken; niet
  teruggedraaid, want de oorzaak was toewijsbaar aan één finding en één regel.
- **2.5** bleek geen zuivere dedup: `detect_mime_type` herkent ook BMP, wat
  `identify_image_format_from_base64` als `image/jpeg` afhandelde en wat geen enkele
  LLM-API accepteert. De delegatie mapt BMP nu expliciet terug, zodat het gedrag
  identiek blijft.

Functionele smoke na afloop (buiten de gate om, omdat de testsuite `images.py` en de
tool-spec-builders niet raakt): alle drie de image-converters, `crop_to_fit`,
`is_image_url`, `identify_image_format_from_base64` inclusief het BMP-pad, en de
tool-schema's van OpenAI-completions, OpenAI-responses en Anthropic geven byte-voor-byte
hetzelfde resultaat als vóór de wijziging.

---

## Applied-log (Modus 1, LIKELY-set)

Alle elf LIKELY-findings kregen "fixen"; geen enkele werd afgewezen of uitgesteld.
Verificatie-gate na de fix-pass, tegen dezelfde baseline:

| Stap | Uitkomst |
|---|---|
| `ruff check --select F,E9 justai/` | **All checks passed** (eerst rood: zie 2.4 hieronder) |
| `ruff check --select F401,F841,C4,SIM justai/` | **All checks passed** |
| Framework-check | n.v.t., geen Django |
| Testsuite | **54 passed, 1 skipped** — gelijk aan na de SAFE-set |
| Import-smoke (alle modules) | **schoon** |
| Entrypoint-smoke | n.v.t., library-package |

| Finding | Keuze | Status | Wat er is gebeurd |
|---|---|---|---|
| 5.4 | argumenten doorgeven | applied | `prompt()` en `chat()` geven nu `images, return_json` mee; geverifieerd dat `Model('sonar-pro').prompt(..., return_json=True)` een `NotImplementedError` geeft |
| 5.5 | fixen | applied | `api_params['messages'] = api_messages[:-1]` in plaats van de lokale herbinding |
| 5.1 | expliciete ValueError | applied | `_pick_image_api_size` eist nu `size`; geverifieerd op `Model('gpt-image-2')` |
| 3.1 | middenweg | applied | `_validate_json_schema` (8 regels) vervangt `is_valid_json_schema` (70); `$schema`-detectie en pad in de melding blijven, `assert` werd een `raise` |
| 2.2 | logger overal | applied | `_map_anthropic_error()`; de `print()`-regels in `completion()` zijn `logger.error()` geworden |
| 2.4 | hard falen | applied | `get_api_key()` in `basemodel.py`, acht providers omgezet; gooit `AuthorizationException` bij een ontbrekende sleutel |
| 2.3 | fixen | applied | `map_openai_error()` in `openai_completions.py`, viermaal gebruikt |
| 4.9 | weghalen | applied | `Message.to_dict` en `Message.from_dict` verwijderd |
| 2.7 | module-functie | applied | `tiktoken_token_count(model_name, text)`, beide klassen delegeren |
| 5.3 | close() implementeren | applied | `BaseModel.close()` sluit de client van elke provider die er een heeft |
| 7.1 | images doorgeven | applied | `Model.prompt_async` normaliseert en geeft `images` door, net als `chat_async` |

**Netto deze pass: 316 regels weg, 163 erbij, over 14 bestanden.**

Twee dingen die tijdens het toepassen anders liepen dan het rapport voorspelde:

- **2.4** liet de gate rood achter met `F841 api_key is assigned but never used` in
  `openai_responses.py`. Dat was geen fout in mijn wijziging maar een bestaande bug die
  pas zichtbaar werd toen de helper de toekenning op één regel zette: de client werd
  aangemaakt zónder `api_key=`. Zie bijvangst 2.4b. Opgelost, niets teruggedraaid.
- **2.2 en 2.3** vroegen allebei een tak die het rapport wel noemde maar die makkelijk te
  missen was: `completion()` gooit in beide bestanden zelf een `NotImplementedError`
  binnen het `try`-blok. Zonder een expliciete `except NotImplementedError: raise` vóór de
  algemene tak was die bewuste melding een `GeneralException` geworden.

Functionele smoke na afloop, buiten de gate om: beide foutmappers over vier
exceptietypen, `get_api_key` voor alle drie de bronnen plus het ontbrekende geval,
`tiktoken_token_count` voor een bekend en een onbekend model, `_validate_json_schema`
voor een geldig en een ongeldig schema, de `size`-guard, de `raise_for_unsupported`-guard
en de context manager. Alle uitkomsten zoals bedoeld.

---

## Nog te doen

1. **7.2** [RISKY] — de doorgeef-façade op `Model` (`stream`, `format_tool_result`,
   `format_assistant_message`, `token_count`). `Agent` omzeilt die façade en gaat
   rechtstreeks naar `self.model.model.…`. Of de façade dun mag blijven of dat `Agent`'s
   directe toegang de fout is, is een ontwerpbeslissing over het publieke oppervlak.
   Blijft handwerk; deze skill past RISKY nooit toe en legt hem ook niet interactief voor.

Verder niets open: alle SAFE- en LIKELY-findings zijn toegepast en geverifieerd.
