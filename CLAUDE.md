# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Build and publish**: `./publish.sh` (bumps version, builds package, uploads to PyPI, commits and tags)
**Run tests**: Run individual test files in tests/ directory (no unified test runner)
**Install dependencies**: `pip install -r requirements.txt`

## Architecture Overview

JustAI is a Python package that provides a unified interface for working with multiple Large Language Model providers through a single `Model` class.

### Core Structure

- **justai/model/model.py**: Main `Model` class - entry point for all LLM interactions
- **justai/models/modelfactory.py**: Factory pattern implementation that routes to appropriate provider based on model name
- **justai/models/**: Provider-specific implementations (OpenAI, Anthropic, Google, etc.)
- **justai/tools/**: Utility modules for caching, image processing, and prompt management

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

### Development Notes

The codebase uses a plugin-like architecture where new providers can be added by:
1. Creating a new model class in justai/models/ inheriting from BaseModel
2. Adding the model name pattern matching logic to ModelFactory
3. Implementing required methods: chat(), chat_async(), generate_image() if supported