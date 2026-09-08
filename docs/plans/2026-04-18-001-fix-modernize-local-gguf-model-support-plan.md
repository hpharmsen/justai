---
title: "fix: Modernize local GGUF model support"
type: fix
status: completed
date: 2026-04-18
---

# Fix: Modernize local GGUF model support

## Overview

De `GuffModel` klasse (`justai/models/gguf_models.py`) is volledig kapot ten opzichte van de huidige `BaseModel` interface. De code is geschreven voor een oudere versie van zowel JustAI als `llama-cpp-python` en moet herschreven worden om weer werkend te krijgen, met als primair doel Gemma 4 lokaal te draaien op Apple Silicon.

## Problem Statement

1. **Interface mismatch** -- methode-signaturen wijken af van `BaseModel` (chat neemt `messages` i.p.v. `prompt`)
2. **Ontbrekende methodes** -- `prompt()` en `prompt_async()` zijn niet geimplementeerd
3. **Init bug** -- Llama client wordt aangemaakt voordat params gezet zijn
4. **Verouderde API** -- gebruikt raw completion `self.client(message)` met hardcoded Llama-2 `[INST]` format i.p.v. `create_chat_completion()` met automatische template detectie
5. **Geen streaming** -- `chat_async()` is niet async en doet geen streaming
6. **n_gpu_layers default van 1** -- moet -1 zijn voor Apple Silicon Metal

## Proposed Solution

Herschrijf `GuffModel` met de moderne `llama-cpp-python` API, volgend het patroon van `AnthropicModel`:

- Gebruik `create_chat_completion()` met messages-array (auto-detecteert chat template uit GGUF metadata)
- Onderhoud conversatie-geschiedenis met `self.messages` (reset in `prompt()`)
- Wrap synchrone streaming iterator voor async generators
- Pas generation params (temperature) per-call toe, niet alleen bij constructie

## Technical Considerations

### Constructor params vs per-call params

`n_ctx`, `n_gpu_layers`, `n_threads`, `n_batch` zijn Llama constructor args -- alleen te wijzigen bij init. `temperature`, `top_p`, `max_tokens` zijn per-call params voor `create_chat_completion()` en moeten bij elke aanroep uit `model_params` gelezen worden zodat runtime-wijzigingen effect hebben.

### Async streaming

`llama-cpp-python` is synchronous (C++ bound). `create_chat_completion(stream=True)` retourneert een synchrone iterator. Voor `prompt_async` en `chat_async` wrappen we dit met `asyncio.get_event_loop().run_in_executor()`.

### Chat template

Moderne GGUF files embedden hun chat template in metadata. `create_chat_completion()` detecteert automatisch het juiste format (Gemma, Llama 3, Mistral, ChatML, etc.). Er is geen handmatige prompt-formatting nodig.

### Model wrapper bug (line 183)

`Model.chat_async()` gebruikt `for word in self.model.chat_async(...)` (sync iteratie) terwijl `prompt_async` en `chat_async_reasoning` `async for` gebruiken. Dit is inconsistent. We fixen dit mee door `chat_async` in Model ook `async for` te maken.

## Acceptance Criteria

- [ ] `GuffModel` implementeert alle `BaseModel` abstract methods met correcte signaturen
- [ ] `Model('/path/to/model.gguf').prompt('Hello')` werkt
- [ ] `Model('/path/to/model.gguf').chat('Hello')` werkt met multi-turn conversatie
- [ ] Streaming werkt via `prompt_async()` en `chat_async()`
- [ ] JSON response format werkt via `return_json=True`
- [ ] `token_count()` geeft accurate telling via `Llama.tokenize()`
- [ ] Default `n_gpu_layers=-1` voor optimale Metal acceleratie
- [ ] Capability flags correct gezet (no images, no tools, yes JSON)
- [ ] Parameters (temperature etc.) zijn runtime-wijzigbaar
- [ ] Werkt met Gemma 4 26B-A4B GGUF op M2 24GB

## Implementation Phases

### Phase 1: Fix GuffModel class

**File: `justai/models/gguf_models.py`** -- volledige herschrijving

```python
class GgufModel(BaseModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f'You are {model_name.split("/")[-1]}, a helpful assistant.'
        super().__init__(model_name, params, system_message)

        # Capability flags
        self.supports_image_input = False
        self.supports_tool_use = False
        self.supports_function_calling = False
        self.supports_return_json = True

        # Constructor params (immutable after init)
        n_ctx = params.get('n_ctx', 8192)
        n_gpu_layers = params.get('n_gpu_layers', -1)
        n_threads = params.get('n_threads', 4)
        n_batch = params.get('n_batch', 512)

        # Per-call params (mutable via model_params)
        self.model_params['temperature'] = params.get('temperature', 0.8)
        self.model_params['max_tokens'] = params.get('max_tokens', 800)

        # Create client AFTER params are set
        self.client = Llama(
            model_path=model_name,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        # Conversation state
        self.messages = []
```

Key decisions:
- Rename `GuffModel` → `GgufModel`
- Set params before creating Llama client
- `n_gpu_layers=-1` default (all layers to Metal GPU)
- Maintain `self.messages` for conversation history
- `max_tokens=800` default (consistent met andere providers, voorkomt truncatie)

**`prompt()`** -- stateless, reset messages:

```python
def prompt(self, prompt, images, tools, return_json, response_format):
    self.messages = []
    return self.chat(prompt, images, tools, return_json, response_format)
```

**`chat()`** -- stateful, bouw messages op:

```python
def chat(self, prompt, images, tools, return_json, response_format):
    self.messages.append({'role': 'user', 'content': prompt})
    messages = [{'role': 'system', 'content': self.system_message}] + self.messages

    kwargs = {
        'messages': messages,
        'temperature': self.model_params.get('temperature', 0.8),
        'max_tokens': self.model_params.get('max_tokens', 800),
    }
    if return_json or response_format:
        kwargs['response_format'] = {'type': 'json_object'}

    output = self.client.create_chat_completion(**kwargs)

    result = output['choices'][0]['message']['content']
    self.messages.append({'role': 'assistant', 'content': result})

    return result, output['usage']['prompt_tokens'], output['usage']['completion_tokens']
```

**`prompt_async()` en `chat_async()`** -- streaming met sync-to-async wrapper:

```python
async def prompt_async(self, prompt, images=None):
    self.messages = []
    async for content, reasoning in self.chat_async(prompt, images):
        yield content, reasoning

async def chat_async(self, prompt, images=None):
    self.messages.append({'role': 'user', 'content': prompt})
    messages = [{'role': 'system', 'content': self.system_message}] + self.messages

    stream = self.client.create_chat_completion(
        messages=messages,
        temperature=self.model_params.get('temperature', 0.8),
        max_tokens=self.model_params.get('max_tokens', 800),
        stream=True,
    )

    full_response = ''
    for chunk in stream:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta and delta['content']:
            full_response += delta['content']
            yield delta['content'], ''

    self.messages.append({'role': 'assistant', 'content': full_response})
```

**`token_count()`** -- implementeer met Llama tokenizer:

```python
def token_count(self, text):
    return len(self.client.tokenize(text.encode('utf-8')))
```

### Phase 2: Update ModelFactory

**File: `justai/models/modelfactory.py`**

- Rename import van `GuffModel` naar `GgufModel`

### Phase 3: Fix Model wrapper async bug

**File: `justai/model/model.py` line 183**

- Wijzig `for word in self.model.chat_async(...)` naar `async for word in self.model.chat_async(...)`

### Phase 4: Test met Gemma 4

- Download Gemma 4 26B-A4B GGUF (Q4_K_M quantisatie, ~17GB) van HuggingFace
- Installeer `llama-cpp-python` met Metal: `CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python`
- Test prompt, chat, streaming, en JSON response

## Dependencies & Risks

- **llama-cpp-python Metal build** -- vereist Xcode command line tools en cmake
- **Gemma 4 GGUF beschikbaarheid** -- afhankelijk van community quantisaties (unsloth/bartowski op HuggingFace)
- **Memory** -- 26B-A4B Q4_K_M (~17GB) past in 24GB maar laat weinig ruimte voor grote context windows
- **Geen echte async** -- llama-cpp houdt GIL vast tijdens inferentie, dus streaming is "token-at-a-time" maar niet concurrent

## Gemma 4 op M2 24GB

**Aanbevolen model:** `gemma-4-26B-A4B-it` (Mixture-of-Experts, ~4B active params)

| Quantisatie | Grootte | Past in 24GB? | Kwaliteit |
|---|---|---|---|
| Q4_K_M | 16.9 GB | Ja | Goed (sweet spot) |
| Q5_K_M | 21.2 GB | Krap | Beter |
| Q3_K_M | 12.5 GB | Ruim | Redelijk |

**Installatie stappen:**
```bash
# 1. Installeer llama-cpp-python met Metal
CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python

# 2. Download model (voorbeeld)
huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF \
  --include "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" \
  --local-dir ./models/

# 3. Gebruik
from justai import Model
m = Model('./models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf', n_ctx=4096)
print(m.prompt('Wat is de hoofdstad van Nederland?'))
```

## Sources & References

- llama-cpp-python docs: https://llama-cpp-python.readthedocs.io/
- llama-cpp-python chat completion API: https://deepwiki.com/abetlen/llama-cpp-python/4.3-chat-completion
- Gemma 4 GGUF: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF
- Apple Silicon Metal install: https://llama-cpp-python.readthedocs.io/en/latest/install/macos/
- Referentie-implementatie: `justai/models/anthropic_models.py`
