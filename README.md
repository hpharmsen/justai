# JustAI

Package to make working with Large Language Models in Python super easy.
Supports OpenAI, Anthropic Claude, Google Gemini, X Grok, DeepSeek, Perplexity, Reve, OpenRouter, Kimi (Moonshot), MiniMax and local GGUF models.

Author: Hans-Peter Harmsen (hp@harmsen.nl) \
Current version: 5.7.2

## Installation
1. Install the package:
```bash
pip install justai
```
2. Create an API key for the provider(s) you intend to use:
   - OpenAI: [platform.openai.com](https://platform.openai.com/account/api-keys)
   - Anthropic: [console.anthropic.com](https://console.anthropic.com/settings/keys)
   - Google: [aistudio.google.com](https://aistudio.google.com/app/apikey)
   - X AI: [console.x.ai](https://console.x.ai)
   - DeepSeek: [platform.deepseek.com](https://platform.deepseek.com)
   - Moonshot: [platform.moonshot.ai](https://platform.moonshot.ai/)
   - MiniMax: [platform.minimax.io](https://platform.minimax.io/)

3. Create a `.env` file with the relevant keys:
```bash
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
X_API_KEY=your-x-ai-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
PERPLEXITY_API_KEY=your-perplexity-api-key
MOONSHOT_API_KEY=your-moonshot-api-key
MINIMAX_API_KEY=your-minimax-api-key
```

## Basic usage

```python
from justai import Model

model = Model('gpt-5-mini')
model.system = """You are a movie critic. I feed you with movie
                  titles and you give me a review in 50 words."""

response = model.chat("Forrest Gump", cached=True)
print(response)
```
The `cached=True` parameter tells justai to cache the prompt and response locally.

## Models

The provider is chosen automatically based on the model name prefix:

| Prefix | Provider |
|--------|----------|
| `gpt*`, `o1*`, `o3*` | OpenAI |
| `claude*` | Anthropic |
| `gemini*` | Google |
| `grok*` | X AI |
| `deepseek*` | DeepSeek |
| `sonar*` | Perplexity |
| `reve*` | Reve |
| `openrouter/*` | OpenRouter |
| `kimi*`, `moonshot*` | Moonshot |
| `minimax*` (case-insensitive) | MiniMax |
| `jev*` | TypeSafe (System One) |
| `systemone/*` | Self-hosted System One |
| `*.gguf` | Local GGUF |

## Features

### JSON and structured output
```python
model = Model('gemini-2.5-flash')
prompt = 'Give me the main characters from Seinfeld. Return json with keys name, profession and weirdness'
data = model.chat(prompt, return_json=True)
```

For typed structured output, pass a Pydantic model or Python type as `response_format`:
```python
from pydantic import BaseModel as PydanticModel

class Character(PydanticModel):
    name: str
    profession: str
    weirdness: str

result = model.chat(prompt, response_format=list[Character])
```

With a Pydantic class, `validation_retries` sends validation errors back to the model and asks
for a corrected answer, up to that many times. Malformed JSON counts as a validation error.
When the answer is still invalid, `ValidationRetryError` (a `ValueError`) is raised. The default
`0` raises the original Pydantic `ValidationError`. Token counters include every attempt.
Validation happens in `Model` for every provider, so a failing validator raises Pydantic's
`ValidationError` (before 5.8 the Anthropic and OpenAI providers wrapped it in `GeneralException`).
```python
from pydantic import field_validator
from justai import ValidationRetryError

class Person(PydanticModel):
    name: str

    @field_validator('name')
    @classmethod
    def full_name(cls, v: str) -> str:
        if ' ' not in v:
            raise ValueError('Must contain first and last name')
        return v

result = model.chat('Who wrote Hamlet?', response_format=Person, validation_retries=2)
```

### Classification (System One models)

System One models do not generate text. They read a state, answer typed questions about
it, and return a calibrated probability for every possible answer. They reply in tens to
hundreds of milliseconds, which makes them practical for routing and triage where a chat
model is too slow and too vague.

A dict of options gives a **choice**:
```python
model = Model('openrouter/typesafe/jev-1.13')
model.classify('My payouts have been failing for 3 days',
               {'payments': 'money and payouts', 'frontend': 'UI', 'account': 'logging in'},
               instructions='Which team should pick this up?')
# {'type': 'choice', 'choice': 'payments', 'confidence': 1.0,
#  'probabilities': {'payments': 1.0, 'frontend': 0.0, 'account': 0.0}}
```

An ordered list of levels gives a **score**. `legend` and `probabilities` are keyed by
level number:
```python
model.classify(ticket, ['can wait', 'normal', 'now'], instructions='How urgent?')
# {'type': 'score', 'score': 1.99, 'confidence': 0.99,
#  'probabilities': {0: 0.0, 1: 0.01, 2: 0.99},
#  'legend': {0: 'can wait', 1: 'normal', 2: 'now'}}
```

No options gives a **noul**, a single yes/no probability:
```python
model.classify(ticket, instructions='Is this a software bug?')
# {'type': 'noul', 'noul': 0.97}
```

Ask several questions in one call with `questions=`; the result is then keyed by question
name:
```python
model.classify(ticket, questions={
    'team': {'type': 'choice', 'instructions': 'Which team?',
             'criteria': {'payments': '...', 'frontend': '...'}},
    'urgency': {'type': 'score', 'instructions': 'How urgent?',
                'criteria': ['low', 'medium', 'high']}})
```

`instructions` is what the model is actually asked, and is required for every question.
Models that are not System One models raise `NotImplementedError`.

### Images
Pass images as URLs, raw bytes or PIL images:
```python
model = Model('gpt-5-nano')
url = 'https://upload.wikimedia.org/wikipedia/commons/9/94/Common_dolphin.jpg'
message = model.chat("What is in this image", images=url)
```

### Image generation
```python
model = Model('gpt-5')
pil_image = model.generate_image("A dolphin reading a book")
```

Input images can be passed for editing or style transfer:
```python
model = Model('gemini-2.5-flash-image-preview')
pil_image = model.generate_image("Convert to Van Gogh style", images=source_image)
```

### Async streaming
```python
import asyncio

async def stream(model_name, prompt):
    model = Model(model_name)
    async for word in model.chat_async(prompt):
        print(word, end='')

asyncio.run(stream('sonar-pro', 'Give me 5 names for a juice bar'))
```

### Prompt caching (Anthropic)

**On by default.** JustAI marks two cache breakpoints on every Anthropic request: one after the system prompt, which also covers the tool definitions, and one after the last turn of the conversation. Multi-turn chats and agent runs reuse their prefix instead of paying full price for it on every call.

The second breakpoint is only set from the second message on. A cache write costs 1.25x, so on a genuine one-shot call there would be nothing to earn it back.

`cached_prompt` still has a job: it moves a large fixed text into the cached prefix, ahead of the varying question.

```python
model = Model('claude-sonnet-4-6')
model.system_message = 'You are an experienced book analyzer'
model.cached_prompt = SOME_LONG_TEXT
response = model.chat('Who is the main character?', cached=False)
```

Two settings, both optional:

```python
model = Model('claude-sonnet-4-6', cache_ttl='1h')       # default '5m'
model = Model('claude-sonnet-4-6', prompt_cache=False)   # no breakpoints at all
```

A 1-hour write costs 2x rather than 1.25x and needs three requests to break even, so it is worth it only when your traffic has gaps longer than five minutes.

**Below the minimum, caching silently does nothing.** The shortest cacheable prefix is 1024 tokens on Sonnet 5, Sonnet 4.6 and Opus 4.8, 512 on Opus 5, 2048 on Opus 4.7, and 4096 on Opus 4.6 and Haiku 4.5. Under that the API caches nothing, without an error and without charging you for it. Zero counters on a short prompt are expected, not a bug.

### Cache counters

```python
model.cache_read_input_tokens      # served from cache, ~0.1x price
model.cache_creation_input_tokens  # written to cache, ~1.25x price
```

Available on every provider. Anthropic reports both; OpenAI and Gemini cache server-side on their own and report reads only. Providers that report nothing stay at zero, which means "not measured" rather than "no cache".

**The two families count differently.** On OpenAI and Gemini the cached tokens are a subset of the input tokens. On Anthropic they sit beside them, so the full prompt size is `input_token_count + cache_read_input_tokens + cache_creation_input_tokens`. If an agent ran for an hour and `input_token_count` reads 4000, the rest came from cache: check the sum, not the single field.

### Effort

Control reasoning depth with a single portable setting. The library translates it to each provider's native parameter.

```python
model = Model('claude-fable-5', effort='low')
# or
model = Model('gpt-5.6-terra')
model.effort = 'xhigh'
```

Valid values: `'low'`, `'medium'`, `'high'`, `'xhigh'`, `'max'`, or `None` (default — send nothing, provider default applies).

**`None` vs `'none'`** — two different things:
```python
Model('gpt-5.6-sol', effort=None)    # default, no reasoning field sent
Model('gpt-5.6-sol', effort='none')  # explicitly turn reasoning off (GPT-5.6 only)
```
`'none'` as a string is a pass-through only accepted by GPT-5.6 models. On every other provider it raises `ValueError`.

| Provider | Native support |
|---|---|
| Anthropic (Fable 5, Mythos 5, Opus 4.7/4.8, Sonnet 5) | full set (`low`/`medium`/`high`/`xhigh`/`max`) |
| Anthropic (Opus 4.6, Sonnet 4.6) | `xhigh` maps up to `max` with warning |
| Anthropic (Opus 4.5) | `xhigh` and `max` map down to `high` with warning |
| OpenAI (`gpt-5.6-*`) | full set; `max` maps to `xhigh` (SDK cap) with warning; also accepts `'none'` |
| Google (Gemini 3.x) | `low`/`medium`/`high`; `xhigh`/`max` map to `HIGH` with warning |
| xAI (`grok-4.5`, `grok-4.3`, `grok-4.20-multi-agent`) | `low`/`medium`/`high`; `xhigh`/`max` map to `high` with warning |
| OpenRouter | passed through raw; OpenRouter maps server-side |
| Older Anthropic (Sonnet 4.5, Haiku 4.5), older OpenAI (o1, o3), Gemini 2.x, DeepSeek, Perplexity, Reve, GGUF | ignored with a warning (dedup'd per Model instance) |

Downmap warnings use a dedicated `EffortDownmapWarning` category so you can filter them:
```python
import warnings
from justai import EffortDownmapWarning

warnings.filterwarnings('ignore', category=EffortDownmapWarning)
# or promote to an error for strict pipelines:
warnings.filterwarnings('error', category=EffortDownmapWarning)
```

**Effort is captured at request initiation.** Mutating `model.effort` during an in-flight `chat_async` does not affect that request.

**Related knobs to consider when raising effort:**
- Anthropic reasoning tokens count against `max_tokens`. On `effort >= 'high'` justai auto-raises `max_tokens` to `4096` if you did not set it explicitly (emits a `UserWarning`). Set `max_tokens` yourself to disable.
- For `effort='xhigh'` or `'max'` on any provider, the default `timeout=120` seconds is often insufficient. Set `timeout=300` (or higher) explicitly.

Level names are **not calibrated across providers** — `'high'` on Anthropic burns different tokens than `'high'` on OpenAI. Re-test cost/latency when switching models.

**A failed call still reports its tokens.** When the provider returns a response that justai then rejects — no text block because the whole budget went to thinking, JSON that will not parse, output truncated at the ceiling — the tokens were billed. `last_token_count()` reports them after the exception, so a caller adding up what a run cost does not lose that spend:

```python
try:
    model.prompt('...')
except BadRequestException:
    spent = model.last_token_count()   # (input, output, total) of the failed call
```

When nothing came back at all (connection error, rate limit), the counters read `(0, 0, 0)` rather than the previous call's numbers.

## Agent

JustAI includes an `Agent` class for autonomous, tool-using agent execution. The agent runs in a loop: it reads a task file, calls tools as needed, and returns a final answer.

### Basic agent usage
```python
import asyncio
from justai import Agent, FileSystemTool

agent = Agent(
    model='claude-sonnet-4-6',
    role='Code reviewer',
    goal='Review Python files and report issues',
    tools=[FileSystemTool(read=['/path/to/src'])],
    max_iterations=10,
)

async def main():
    async for event in agent.run('tasks.md'):
        if event.type == 'response':
            print(event.content, end='')
        elif event.type == 'done':
            print(f'\nAnswer: {event.result.answer}')

asyncio.run(main())
```

### Built-in tools

**FileSystemTool** — read/write files with path traversal protection:
```python
FileSystemTool(read=['/allowed/read/dir'], write=['/allowed/write/dir'])
```

**ShellTool** — run shell commands with allowlist-based security:
```python
ShellTool(allowlist=['echo', 'ls', 'python'])
```

**WebFetchTool** — fetch URLs with SSRF protection:
```python
WebFetchTool()
```

### Custom tools
```python
@agent.tool
def search_database(ctx, query: str) -> str:
    """Search the database for matching records."""
    return db.search(query)
```

Tool arguments are validated against the tool's signature before the tool runs. Values are
coerced where Pydantic allows it (`"12"` for an `int` arrives as `12`), a parameter typed as a
Pydantic model arrives as an instance, and unknown arguments are rejected. Invalid or malformed
arguments go back to the model as the tool result, with the errors, and the tool does not run.
Each tool gets `validation_retries` attempts in a row (default 2). After that the run stops with an
`error` event plus `done`, and `AgentResult.error` says why.
```python
agent = Agent('claude-sonnet-5', tools=[create_invoice], validation_retries=2)
```

### Dynamic instructions
```python
@agent.instructions
def inject_context(ctx) -> str:
    return f'Current user: {ctx.deps["username"]}'
```

### Skills
Load `.md` skill files to extend the agent's system prompt:
```python
agent = Agent(
    model='claude-sonnet-4-6',
    role='Assistant',
    goal='Help with tasks',
    skills_dir='./skills',
)
```

### Agent events
The `agent.run()` async generator yields `AgentEvent` objects with these types:
- `status` — status messages
- `response` — streamed text from the model
- `tool_call` — tool invocation (with `name`, `arguments`, `tool_result`)
- `error`: the run stopped (provider failure or exhausted validation retries); `done` follows
- `done` — final result with `AgentResult` (answer, audit trail, token usage, iterations)

## License

MIT
