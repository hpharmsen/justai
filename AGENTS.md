# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

**Build and publish**: `./publish.sh` (bumps version, builds package, uploads to PyPI, commits and tags)
**Run tests**: Run individual test files in tests/ directory (no unified test runner)
**Install dependencies**: `uv sync` (or `uv sync --extra llama` for local GGUF model support)

## Architecture Overview

JustAI is a Python package that provides a unified interface for working with multiple Large Language Model providers through a single `Model` class.

### Core Structure

- **justai/model/model.py**: Main `Model` class - entry point for all LLM interactions
- **justai/models/modelfactory.py**: Factory pattern implementation that routes to appropriate provider based on model name
- **justai/models/**: Provider-specific implementations (OpenAI, Anthropic, Google, etc.)
- **justai/tools/**: Utility modules for caching, image processing, and prompt management
- **docs/solutions/**: documented solutions to past problems (Dutch, flat, frontmatter `title`, `date`, `tags`, `files`); relevant when working on the files a doc lists

### Key Design Patterns

**Model Factory**: The `ModelFactory.create()` method uses model name prefixes to determine the appropriate provider:
- `gpt*`, `o1*`, `o3*` → OpenAI (via openai_responses.py)
- `claude*` → Anthropic
- `gemini*` → Google
- `grok*` → X AI
- `deepseek*` → DeepSeek
- `sonar*` → Perplexity
- `reve*` → Reve
- `openrouter/*` → OpenRouter
- `kimi*`, `moonshot*` → Moonshot
- `minimax*` (case-insensitive) → MiniMax
- `jev*` → TypeSafe System One
- `systemone/*` → Self-hosted System One
- `*.gguf` → Local GGUF models

**Base Model Pattern**: All provider implementations inherit from `BaseModel` in basemodel.py which defines the common interface.

**Message System**: Conversations are managed through `Message` objects that maintain chat history and context.

**Caching**: Built-in response caching system in tools/cache.py for improved performance and cost reduction.

### Key Features Architecture

- **Multi-modal support**: Images can be passed as PIL images, URLs, or raw data
- **Async support**: Streaming responses via `chat_async()` methods
- **Tool/Function calling**: Support for provider-specific function calling capabilities
- **Prompt caching**: Provider-specific prompt caching (e.g., Anthropic's prompt caching)
- **JSON responses**: Structured output support with `return_json=True` parameter
- **Validation retries**: `justai/tools/validation.py` holds `RepairBudget`, the error formatter and `ValidationRetryError`. `Model.chat`/`prompt(validation_retries=N)` validate a Pydantic `response_format` centrally and send the errors back for a repair (chat via provider history, prompt statelessly with question plus bad answer). Anthropic and OpenAI return raw JSON text for Pydantic formats instead of using the SDK `parse()` calls. The `Agent` validates every tool call against a strict args model before running it (`validation_retries`, default 2); malformed tool JSON arrives as `ToolCallRequest.raw_arguments`. Every error that stops an Agent run yields an `error` event plus `done` and fills `AgentResult.error`.
- **Classification**: `Model.classify()` for System One models, which return typed decisions with calibrated probabilities instead of text. One wire format (`POST /v1/systemone`) behind three entrances: `jev*` at TypeSafe, `openrouter/typesafe/jev-*` at OpenRouter, and `systemone/<name>` at any self-hosted server. `systemone.py` holds the format, `SystemOneMixin` the HTTP half with its own keep-alive client and 429/529 retry loop. `instructions` is required per question; score answers key `legend` and `probabilities` by int level, matching the vendor SDKs. Every other model raises `NotImplementedError`.

### Development Notes

The codebase uses a plugin-like architecture where new providers can be added by:
1. Creating a new model class in justai/models/ inheriting from BaseModel
2. Adding the model name pattern matching logic to ModelFactory
3. Implementing required methods: chat(), chat_async(), generate_image() if supported