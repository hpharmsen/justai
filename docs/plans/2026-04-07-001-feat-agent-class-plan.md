---
title: "feat: Add Agent class for autonomous agent execution"
type: feat
status: active
date: 2026-04-07
origin: docs/ideation/2026-04-07-agent-class-ideation.md
---

# feat: Add Agent class for autonomous agent execution

## Overview

Add an `Agent` class to JustAI that provides autonomous agent execution with skills (`.md` files), tool-calling, dependency injection, and a REPL loop driven by `tasks.md`. The Agent wraps the existing `Model` class (composition, not inheritance) and adds: skill loading, tool registration, an agentic execution loop with streaming output, error recovery, and an audit trail.

## Problem Statement / Motivation

JustAI currently provides excellent low-level LLM access via the `Model` class, but building an autonomous agent on top requires significant boilerplate: managing a conversation loop, registering tools, handling errors, tracking task state, and composing system prompts from multiple sources.

The Agent class captures these patterns into a clean, reusable abstraction. Our research of 7 existing frameworks (LangChain, LlamaIndex, AutoGen, CrewAI, smolagents, OpenAI Agents SDK, PydanticAI) revealed that all over-engineer the common case. JustAI's Agent can be simpler while offering unique capabilities: `.md` skill files, self-modifying `tasks.md`, and permissions built into tool objects.

## Proposed Solution

### API Design

```python
from justai import Agent, FileSystemTool, ShellTool, WebFetchTool

agent = Agent(
    model='claude-sonnet-4-6',          # or an existing Model instance
    role='Senior Python developer',     # optional
    goal='Fix all failing tests',       # optional
    skills_dir='./skills',
    tools=[
        FileSystemTool(read=['./src'], write=['./output']),
        ShellTool(allowlist=['git', 'pytest']),
        WebFetchTool(),
        my_standalone_function,          # raw callables accepted
    ],
    max_retries=3,
    max_iterations=50,
    verbose=True,                        # False for brief status updates only
)

@agent.tool
def query_db(ctx, sql: str) -> str:
    """Run a read-only SQL query."""
    return ctx.deps.db.fetch(sql)

@agent.instructions
def inject_context(ctx) -> str:
    return f'Current user: {ctx.deps.user_id}'

# Streaming usage — async generator yields events as the agent runs
async for event in agent.run('tasks.md', deps={'db': db, 'user_id': '123'}):
    if event.type == 'status':
        print(event.message)           # "Starting task: fix test_login"
    elif event.type == 'tool_call':
        print(f"Tool: {event.name}")   # "Tool: read_file"
    elif event.type == 'response':
        print(event.content, end='')   # LLM text, token by token
    elif event.type == 'done':
        result = event.result          # AgentResult

# Simple usage — collect final result only
result = await agent.run_until_done('tasks.md', deps={'db': db, 'user_id': '123'})
```

### Key Design Decisions

- **Async generator `run()`**: `agent.run()` is an async generator that yields `AgentEvent` objects as the agent progresses. This gives callers real-time visibility into the agent's work — streaming LLM text, tool calls, status updates.
- **`run_until_done()` convenience**: For callers who just want the final result without streaming, `run_until_done()` consumes the generator and returns `AgentResult`.
- **Agent manages conversation history**: The Agent maintains its own message list. It uses a new `Model.stream()` method for LLM calls — a stateless method that accepts messages and streams responses. This gives the Agent full control over context (can summarize, truncate, etc. in v2). Existing `Model.chat()` and `Model.prompt()` are **not modified**.
- **`Model.stream()` keeps provider abstraction in Model**: The new method normalizes provider-specific response formats (Anthropic tool_use blocks, OpenAI function_calls) into a common `StreamChunk` type. No provider-specific code in Agent.
- **Simple `deps`**: deps is any object (dict, dataclass, whatever). No generics, no `deps_type` validation. Tools receive it via `ctx.deps`.
- **`ctx` object**: A simple namespace with `ctx.deps` and `ctx.agent`. No Generic type parameter.
- **`@agent.instructions` evaluated once per `run()` call**, not per iteration.
- **Skills**: All `.md` files in `skills_dir` concatenated into system prompt (alphabetical). No `{{include}}` in v1.
- **No auto-discovery from directories**: `@agent.tool` + raw callables cover the use case. Deferred to v2.
- **Providers**: v1 supports Anthropic + OpenAI only. Others added incrementally.

### Event Types

```python
@dataclass
class AgentEvent:
    type: str        # 'status' | 'response' | 'tool_call' | 'task_update' | 'error' | 'done'
    message: str | None = None       # for 'status', 'error'
    content: str | None = None       # for 'response' (streaming text chunk)
    name: str | None = None          # for 'tool_call' (tool name)
    arguments: dict | None = None    # for 'tool_call'
    tool_result: str | None = None   # for 'tool_call' (after execution)
    result: AgentResult | None = None  # for 'done'
```

### Return Type

```python
@dataclass
class AgentResult:
    answer: str                          # what the agent passed to final_answer()
    audit: list[AuditEntry]             # structured log of all tool calls
    tasks: str                           # final tasks.md content
    tokens: tuple[int, int]             # (total_input, total_output)
    iterations: int                      # how many loop iterations ran

@dataclass
class AuditEntry:
    timestamp: str
    tool_name: str
    arguments: dict
    result: str
    duration_ms: int
    success: bool
```

## Technical Approach

### Architecture

```
┌─────────────────────────────────────────────────┐
│                     Agent                       │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐ │
│  │ Skills   │  │ Tools    │  │ Context        │ │
│  │ Loader   │  │ Registry │  │ (deps)         │ │
│  └────┬─────┘  └────┬─────┘  └───────┬────────┘ │
│       │             │                │          │
│       ▼             ▼                ▼          │
│  ┌──────────────────────────────────────────┐   │
│  │        Agent Loop (async generator)      │   │
│  │  Agent manages own message history       │   │
│  │  1. Compose system prompt                │   │
│  │  2. Call Model.stream(messages, tools)   │   │
│  │  3. Yield streaming text to caller       │   │
│  │  4. Execute tool calls (sequential)      │   │
│  │  5. Yield tool call events to caller     │   │
│  │  6. Add tool results to messages         │   │
│  │  7. Log to audit trail                   │   │
│  │  8. Update tasks.md                      │   │
│  │  9. Loop until final_answer() or max     │   │
│  └──────────────────────────────────────────┘   │
│       │                                         │
│       ▼                                         │
│  ┌──────────┐                                   │
│  │  Model   │  (existing — only stream() added) │
│  └──────────┘                                   │
└─────────────────────────────────────────────────┘
```

### Critical: `Model.stream()` Architecture

Instead of modifying existing `Model.chat()` or `Model.prompt()`, we add a single new method: `Model.stream()`. This is a **stateless** method — the caller (Agent) passes in the full conversation and gets back a streaming response.

**New types** in `justai/models/basemodel.py`:

```python
@dataclass
class ToolCallRequest:
    id: str              # provider-specific call ID
    name: str            # tool/function name
    arguments: dict      # parsed arguments

@dataclass
class StreamChunk:
    type: str            # 'text' | 'tool_calls' | 'done'
    content: str | None = None              # for 'text'
    tool_calls: list[ToolCallRequest] | None = None  # for 'tool_calls'
    input_tokens: int | None = None         # for 'done'
    output_tokens: int | None = None        # for 'done'
```

**New method** on `Model`:

```python
async def stream(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncGenerator[StreamChunk, None]:
    """Stateless streaming call. Caller manages conversation history.

    messages: list of dicts with 'role' and 'content' keys
    tools: list of tool specs (same format as Model.tools)

    Yields StreamChunk objects:
    - type='text': streaming text content
    - type='tool_calls': LLM wants to call tools (list of ToolCallRequest)
    - type='done': final chunk with token counts
    """
    async for chunk in self.model.stream(messages, tools):
        yield chunk
```

**Provider implementations** (new abstract method on `BaseModel`):

Each provider implements `async def stream(messages, tools) -> AsyncGenerator[StreamChunk]`:

- **AnthropicModel**: Uses `client.messages.stream()` API. Yields `StreamChunk(type='text')` for text deltas. When `tool_use` blocks appear, collects them and yields `StreamChunk(type='tool_calls', tool_calls=[...])`.
- **OpenAIResponsesModel**: Uses streaming Responses API. Yields text chunks. Maps `function_call` items to `ToolCallRequest` objects. **Fixes existing bug**: handles multiple parallel function_calls (currently variable overwritten in loop).
- Other providers: not implemented in v1, raise `NotImplementedError`.

**What this means for Agent:**

The Agent loop becomes simple:

```python
async def run(self, tasks_file, deps=None):
    self.messages = [{'role': 'system', 'content': system_prompt}]
    self.messages.append({'role': 'user', 'content': tasks_content})

    for iteration in range(self.max_iterations):
        response_text = ''
        tool_calls = []

        async for chunk in self.model.stream(self.messages, self.tools):
            if chunk.type == 'text':
                response_text += chunk.content
                yield AgentEvent(type='response', content=chunk.content)
            elif chunk.type == 'tool_calls':
                tool_calls = chunk.tool_calls
            elif chunk.type == 'done':
                self._track_tokens(chunk.input_tokens, chunk.output_tokens)

        self.messages.append({'role': 'assistant', 'content': response_text})

        if tool_calls:
            for tc in tool_calls:
                result = self._execute_tool(tc)
                yield AgentEvent(type='tool_call', name=tc.name,
                                arguments=tc.arguments, tool_result=result)
                self.messages.append(self._format_tool_result(tc, result))
        elif self._is_final_answer:
            yield AgentEvent(type='done', result=self._build_result())
            return

        self._update_tasks_file(tasks_file)
        yield AgentEvent(type='task_update', message=f'Iteration {iteration + 1}')

    yield AgentEvent(type='done', result=self._build_result())  # max_iterations
```

**Key advantages over the previous `return_tool_calls` approach:**

| Aspect | Previous approach | New approach |
|--------|-------------------|-------------|
| Model changes | Modify `chat()` + add `submit_tool_results()` | Add one new `stream()` method |
| Existing methods | Modified | Untouched |
| Conversation history | Model manages (conflict with Agent) | Agent manages (full control) |
| Streaming to caller | Not possible (sync) | Built-in (async generator) |
| Provider abstraction | In Model | In Model (unchanged) |
| Tool result format | New `submit_tool_results()` needed | Agent appends to its own message list |

### Tool Result Message Format

The Agent needs to format tool results in the correct provider-specific format. Rather than baking this into the Agent, `Model` exposes a helper:

```python
def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
    """Format a tool result message for this provider.
    Returns a dict ready to be added to the messages list."""
    return self.model.format_tool_result(tool_call_id, tool_name, result)
```

Each provider implements this:
- **Anthropic**: `{'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': id, 'content': result}]}`
- **OpenAI**: `{'type': 'function_call_output', 'call_id': id, 'output': result}`

This keeps all provider-specific formatting in Model/BaseModel.

### File Structure

```
justai/
  __init__.py                    # Add Agent + tool exports
  agent/
    __init__.py                  # Export Agent, AgentResult, AgentEvent
    agent.py                     # Agent class + AgentContext + AgentResult + AuditEntry + AgentEvent
    skills.py                    # Skill loader (read + concatenate .md files)
    tools/
      __init__.py
      filesystem.py              # FileSystemTool
      shell.py                   # ShellTool
      web_fetch.py               # WebFetchTool
```

7 files in Agent package. Model changes: `stream()` + `format_tool_result()` + `ToolCallRequest` + `StreamChunk`.

### Implementation Phases

#### Phase 1: Foundation — `Model.stream()` + Core Agent Loop

**Goal:** Streaming agent loop running with `@agent.tool` tools.

**Tasks:**

- [ ] Define `ToolCallRequest` and `StreamChunk` dataclasses in `justai/models/basemodel.py`
- [ ] Add abstract `stream(messages, tools)` method to `BaseModel`
- [ ] Add abstract `format_tool_result(tool_call_id, tool_name, result)` method to `BaseModel`
- [ ] Implement `stream()` in `AnthropicModel`:
  - Use `client.messages.stream()` for streaming
  - Yield `StreamChunk(type='text')` for text deltas
  - Collect `tool_use` blocks, yield as `StreamChunk(type='tool_calls')`
  - Yield `StreamChunk(type='done')` with token counts at end
- [ ] Implement `stream()` in `OpenAIResponsesModel`:
  - Use streaming Responses API
  - Fix existing bug: handle multiple parallel function_calls
  - Map function_call items to `ToolCallRequest`
  - Yield text chunks and tool_calls
- [ ] Implement `format_tool_result()` in both providers
- [ ] Add `Model.stream()` and `Model.format_tool_result()` as forwarding methods
- [ ] Create `justai/agent/agent.py`:
  - `Agent.__init__()`: accepts `model` (str or Model), `role`, `goal`, `tools`, `skills_dir`, `max_retries`, `max_iterations`, `verbose`, `model_kwargs`
  - `Agent.run(tasks_file, deps=None)` — async generator:
    1. Read tasks.md
    2. Compose system prompt (role + goal)
    3. Build messages list (Agent-managed)
    4. Call `Model.stream(messages, tools)` — yields streaming chunks
    5. Yield `AgentEvent` objects to caller (text, tool_calls, status)
    6. Execute tool calls, add results to messages via `format_tool_result()`
    7. Check for `final_answer` → yield done event
    8. Update tasks.md (atomic write)
    9. Loop until done or max_iterations
  - `Agent.run_until_done(tasks_file, deps=None)` — convenience, consumes generator, returns `AgentResult`
  - `@agent.tool` decorator: register callable with docstring/type hints as schema
  - `@agent.instructions` decorator: register callable, evaluated once per `run()`
  - `final_answer` registered as built-in tool automatically
  - `AgentEvent`, `AgentResult`, `AuditEntry`, `AgentContext` dataclasses (all in agent.py)
- [ ] Update `justai/__init__.py` to export `Agent`, `AgentResult`, `AgentEvent`
- [ ] Write tests: streaming events, final_answer exits loop, max_iterations works, tool results formatted correctly per provider

**Success criteria:** Agent streams events to caller while running tasks.md to completion via Anthropic or OpenAI.

**Estimated effort:** High (provider stream implementations)

#### Phase 2: Built-in Tools

**Goal:** FileSystemTool, ShellTool, WebFetchTool with proper security.

**Tasks:**

- [ ] Create `justai/agent/tools/filesystem.py` — `FileSystemTool`:
  - Constructor: `FileSystemTool(read=[...], write=[...])`
  - Tools exposed: `read_file(path)`, `write_file(path, content)`, `list_directory(path)`
  - **Path traversal prevention:**
    1. `Path(path).resolve()` to absolute
    2. `resolved.is_relative_to(allowed.resolve())` check
    3. Reject symlinks on write: `path.is_symlink()` check
  - Permission violation: return clear error message to LLM
- [ ] Create `justai/agent/tools/shell.py` — `ShellTool`:
  - Constructor: `ShellTool(allowlist=['git', 'pytest'])`
  - Tool exposed: `run_command(executable: str, args: list[str])` — **NOT a single command string**
  - `subprocess.run(shell=False)` — no shell interpretation
  - Validate: `executable` in allowlist (exact match)
  - Reject: arguments containing shell metacharacters (`;`, `|`, `&&`, `$`, `` ` ``)
  - Return: `{stdout, stderr, exit_code}`
  - Timeout: configurable, default 60 seconds
- [ ] Create `justai/agent/tools/web_fetch.py` — `WebFetchTool`:
  - Tool exposed: `fetch_url(url: str, raw: bool = False)`
  - **SSRF protection:** http/https only, block private IPs, block metadata endpoints
  - **Response size limit:** max 1MB, abort if exceeded
  - **Timeout:** 30 seconds
  - Default: cleaned text via `html.parser` stdlib
  - Backend: `httpx`
- [ ] Each tool class: `get_tools()` → list of `(name, description, parameters, callable)` tuples
- [ ] Write security tests: path traversal, shell injection, SSRF, size limits

**Success criteria:** All three built-in tools work with security enforcement.

**Estimated effort:** Medium

#### Phase 3: Skills + Error Recovery + Polish

**Goal:** Skill loading, dynamic instructions, error recovery, output modes.

**Tasks:**

- [ ] Create `justai/agent/skills.py`:
  - Read all `.md` files from `skills_dir` (alphabetical)
  - Concatenate into single string
  - Raise `FileNotFoundError` if `skills_dir` set but doesn't exist
- [ ] System prompt composition: role/goal → skills → `@agent.instructions` output
- [ ] Error recovery:
  - Tool failure: sanitize error (type + description, no tracebacks), inject to LLM
  - Non-existent tool: inject error listing available tools
  - Validation failure: retry up to `max_retries` with sanitized error
  - `max_retries` exhausted: annotate task as `[FAILED]`, continue
  - `RatelimitException` → wait + retry, `AuthorizationException` → abort
- [ ] tasks.md self-modification:
  - Re-read each iteration, update checkboxes, annotate failures
  - Atomic write: temp file + `os.replace()`
- [ ] Verbose / brief output:
  - `verbose=True`: yield all events (text, tool_calls, status)
  - `verbose=False`: yield only status and done events
- [ ] Token tracking: accumulate after each `stream()` call from done chunks
- [ ] Edge cases: empty tasks.md, no tools, no skills_dir
- [ ] Write integration tests

**Success criteria:** Full agent functionality with error recovery and streaming output.

**Estimated effort:** Medium

## Alternative Approaches Considered

| Approach | Why Rejected |
|----------|-------------|
| `return_tool_calls` on `Model.chat()` | Requires modifying existing methods; two-owner conflict for conversation history. New `stream()` method is cleaner — additive, no breaking changes |
| Agent manages provider-specific formats | Provider abstraction belongs in Model. `Model.stream()` + `format_tool_result()` keep all provider logic in one place |
| Synchronous blocking `run()` | No visibility into agent progress. Async generator yields events in real-time |
| Subclass `Model` instead of wrapping it | Couples Agent to Model internals |
| `RunContext[Deps]` with generics | Framework-ahead-of-need; simple `ctx.deps` is sufficient |
| Auto-discovery from `.py` directories | Import executes all module-level code; security risk |
| `run_shell(command)` as single string | Shell injection risk. argv-style `run_command(executable, args)` is safe |
| WebSearchTool as built-in | Tangential; ship as example |
| `{{include}}` skill syntax in v1 | Flat concatenation is sufficient |

## System-Wide Impact

### Interaction Graph

```
async for event in agent.run(tasks_file, deps):
  → @agent.instructions functions executed (once per run)
  → SkillLoader reads + concatenates .md files
  → System prompt composed (role + goal + skills + instructions)
  → Messages list initialized (Agent-managed)
  → Agent loop starts:
    → Model.stream(messages, tools) called
      → Provider streams response (text chunks + tool calls)
      → Agent yields AgentEvent(type='response') for each text chunk
    → If tool_calls received:
      → Agent executes tools sequentially
      → Agent yields AgentEvent(type='tool_call') for each
      → Agent adds results to messages via Model.format_tool_result()
    → If final_answer tool called:
      → Agent yields AgentEvent(type='done', result=AgentResult)
      → Generator exits
    → tasks.md re-read, updated, atomically written
    → Agent yields AgentEvent(type='task_update')
    → Loop continues until done or max_iterations
```

### Error Propagation

```
Provider API error (RatelimitException)
  → Agent catches → waits → retries automatically
Provider API error (AuthorizationException)
  → Agent catches → yields error event → aborts → raises to caller
Tool execution error
  → Agent catches → sanitizes (no tracebacks) → adds to messages → LLM adjusts
Validation error
  → Agent catches → sanitizes → retries up to max_retries
  → On exhaustion → annotates task as FAILED → continues
Max iterations reached
  → Agent yields done event with partial AgentResult
Tool permission denied
  → Tool returns clear error message → added to messages → LLM adjusts
```

### State Lifecycle Risks

- **tasks.md partial write:** Mitigated by atomic write (temp file + `os.replace()`).
- **Message history growth:** Agent manages its own messages. For v1, accept the risk — context overflow produces a provider error caught by the error handler. Context management (summarization, sliding window) is v2.
- **Audit trail in-memory only:** Lost if process crashes. Acceptable for v1.

### API Surface Parity

New public API surface:
- `Agent` class (constructor + `run()` + `run_until_done()` + `@agent.tool` + `@agent.instructions`)
- `AgentEvent`, `AgentResult`, `AuditEntry` dataclasses
- `FileSystemTool`, `ShellTool`, `WebFetchTool` classes

Changes to existing API (additive only):
- `BaseModel`: new `ToolCallRequest` + `StreamChunk` dataclasses, new abstract `stream()` + `format_tool_result()` methods
- `Model`: forwards `stream()` and `format_tool_result()` to provider
- `AnthropicModel`: implements `stream()` + `format_tool_result()`
- `OpenAIResponsesModel`: implements `stream()` + `format_tool_result()` + fixes parallel function_call bug

**No existing methods modified.** `chat()`, `prompt()`, `chat_async()`, `prompt_async()` are untouched.

### Integration Test Scenarios

1. **Streaming full run:** Agent streams events while running tasks.md — verify text chunks, tool calls, and done event arrive in order
2. **`run_until_done()` convenience:** Same scenario, verify final AgentResult is correct
3. **Tool permission denial:** FileSystemTool write outside allowed path → verify error fed back to LLM
4. **Shell injection prevention:** ShellTool with metacharacters → verify rejection
5. **SSRF blocking:** WebFetchTool to private IP → verify blocked
6. **Error recovery:** Tool fails → LLM retries → verify audit trail shows both attempts
7. **Max iterations:** Agent stuck → max_iterations triggers → verify partial result

## Acceptance Criteria

### Functional Requirements

- [ ] `agent.run()` is an async generator yielding `AgentEvent` objects
- [ ] `agent.run_until_done()` returns `AgentResult` directly
- [ ] Agent streams LLM text responses token-by-token to caller
- [ ] Built-in tools work: FileSystemTool, ShellTool, WebFetchTool
- [ ] `@agent.tool` decorator registers tools with context injection
- [ ] `@agent.instructions` composes dynamic context into system prompt
- [ ] Skills load from `.md` files in `skills_dir` (flat concatenation)
- [ ] `final_answer()` terminates the loop
- [ ] Error recovery: tool failures sanitized and fed back to LLM
- [ ] tasks.md is read/written each iteration (checkbox updates, failure annotations)
- [ ] `verbose=True/False` controls which events are yielded
- [ ] `max_iterations` prevents runaway loops
- [ ] Path traversal, shell injection, and SSRF are prevented
- [ ] Audit trail captures all tool executions

### Non-Functional Requirements

- [ ] Agent wraps Model via composition, no provider-specific code in Agent
- [ ] Works with Anthropic and OpenAI providers (others added incrementally)
- [ ] No new required dependencies (html.parser is stdlib, httpx already a dependency)
- [ ] Existing Model methods (`chat`, `prompt`, `chat_async`, `prompt_async`) untouched
- [ ] Token usage tracked and reported in AgentResult

### Quality Gates

- [ ] Tests for `Model.stream()` on Anthropic and OpenAI
- [ ] Tests for each built-in tool including security enforcement
- [ ] Tests for `@agent.tool` decorator and context injection
- [ ] Tests for streaming event sequence
- [ ] Integration test: full agent run with tasks.md
- [ ] Security tests: path traversal, shell injection, SSRF blocking

## Dependencies & Prerequisites

- **`Model.stream()` method** — Phase 1. New additive method on Model/BaseModel; no existing API changes.
- **httpx** — already a dependency
- **html.parser** — stdlib, no new dependency
- No optional `[agent]` extra needed — no new dependencies

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Provider streaming APIs differ significantly | Medium | High | Each provider implements its own `stream()` mapping to common `StreamChunk` format |
| OpenAI parallel function_call bug | High | Medium | Fix as part of Phase 1 `stream()` implementation |
| Shell command injection | Medium | Critical | argv-style API with `shell=False`; reject metacharacters |
| Path traversal in FileSystemTool | Medium | High | `Path.resolve()` + `is_relative_to()`; reject symlinks on write |
| SSRF via WebFetchTool | Medium | High | Scheme validation + private IP blocklist + DNS resolution check |
| Prompt injection via external content | Medium | High | Sanitize tool output; document trust requirements |
| Context window overflow on long runs | Medium | Medium | Provider error caught by error handler; summarization in v2 |
| Error messages leak internals to LLM | Low | Medium | Sanitize: exception type + generic description only |

## Future Considerations (v2+)

- `needs_approval=True` per tool for human-in-the-loop
- Multi-agent coordination / sub-agent fan-out
- Cost/token budget enforcement with model downgrade
- Persistent audit trail (SQLite)
- Context window management (summarization, sliding window)
- MCP server tool integration
- Auto-discovery from `.py` directories (with AST-based scanning for safety)
- `{{include}}` syntax for skill cross-references with cycle detection
- WebSearchTool as built-in
- Google/other provider support for `stream()`
- Parallel tool execution option

## Sources & References

### Origin

- **Origin document:** [docs/ideation/2026-04-07-agent-class-ideation.md](../ideation/2026-04-07-agent-class-ideation.md) — key decisions: permissions via tool objects, `final_answer()` done signal, `@agent.instructions`, role/goal config, self-modifying tasks.md

### Design Validation

- Reference implementation: `examples/agent_reference.py` — working prototype that validates the API design (async generator, tool registration, permission enforcement, event streaming). Uses prompt engineering for tool calls as workaround; the real implementation requires native provider tool-calling via `Model.stream()`.

### Internal References

- Model class: `justai/model/model.py` (composition target)
- BaseModel interface: `justai/models/basemodel.py` (abstract methods, capability flags)
- Tool registration: `justai/model/model.py:118-126` (existing `add_tool()`)
- Anthropic tool loop: `justai/models/anthropic_models.py` (completion method)
- OpenAI tool loop: `justai/models/openai_responses.py` (prompt method, parallel function_call bug)
- Message system: `justai/model/message.py` (Message, ToolUseMessage)
- Existing async: `Model.prompt_async()`, `Model.chat_async()` (streaming pattern reference)

### External References

- PydanticAI docs: agent-scoped tools, context injection, @agent.instructions
- smolagents docs: final_answer() termination pattern
- CrewAI docs: role/goal agent identity

### Research & Review

- Framework comparison of 7 agent libraries conducted 2026-04-07
- Document reviewed by coherence, feasibility, security, scope-guardian personas
- Architecture revised: `Model.stream()` replaces `return_tool_calls` approach — additive, no breaking changes, streaming output, Agent-managed conversation history
