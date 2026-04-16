# JustAI

Package to make working with Large Language Models in Python super easy.
Supports OpenAI, Anthropic Claude, Google Gemini, X Grok, DeepSeek, Perplexity, Reve, OpenRouter and local GGUF models.

Author: Hans-Peter Harmsen (hp@harmsen.nl) \
Current version: 5.5.2

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

3. Create a `.env` file with the relevant keys:
```bash
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
X_API_KEY=your-x-ai-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
PERPLEXITY_API_KEY=your-perplexity-api-key
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
```python
model = Model('claude-sonnet-4-6')
model.system_message = 'You are an experienced book analyzer'
model.cached_prompt = SOME_LONG_TEXT
response = model.chat('Who is the main character?', cached=False)
```

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
- `error` — error messages
- `done` — final result with `AgentResult` (answer, audit trail, token usage, iterations)

## License

MIT
