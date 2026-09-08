---
date: 2026-04-07
topic: agent-class
focus: Adding an Agent class to JustAI for autonomous agent development
---

# Ideation: Agent Class for JustAI

## Codebase Context
JustAI is a Python package (v5.5.1) with a unified `Model` class routing to 8+ LLM providers via a factory pattern. It has provider-specific tool-calling, SQLite caching, async streaming, and a Message-based conversation system. No agent abstraction exists yet. The proposed `Agent` class sits naturally alongside `justai/models/` and `justai/tools/`.

## Proposed Agent Design (refined after research)

```python
@dataclass
class AppDeps:
    db: Database

agent = Agent(
    model="claude-sonnet-4-6",
    role="Senior developer",
    goal="Fix all failing tests",
    skills_dir="./skills",
    deps_type=AppDeps,
    tools=[
        FileSystemTool(read=["./src"], write=["./output"]),
        ShellTool(allowlist=["git", "pytest"]),
        WebFetchTool(),
        WebSearchTool(),
        "./custom_tools/",       # auto-discovers .py files
        my_custom_function,      # or individual callables
    ],
    max_retries=3,
)

@agent.tool
async def query_db(ctx: RunContext[AppDeps], sql: str) -> str:
    """Run a read-only SQL query."""
    return await ctx.deps.db.fetch(sql)

@agent.instructions
async def inject_context(ctx: RunContext[AppDeps]) -> str:
    return f"Current schema: {await ctx.deps.db.schema()}"

result = await agent.run("tasks.md", deps=AppDeps(db=db))
```

### Key design decisions settled
- **Permissions via tools**: no separate permission API; `FileSystemTool` and `ShellTool` carry their own constraints
- **Mixed tool list**: accepts tool objects, directory paths (auto-discovery), or raw callables
- **`@agent.tool` decorator**: for inline tool definitions with typed dependency injection (PydanticAI pattern)
- **`@agent.instructions` decorator**: for dynamic prompt fragments that compose with static .md skills
- **Typed `deps`**: dataclass injected into tools via `RunContext` — no globals, testable
- **`role` / `goal`**: simple identity config auto-injected into system prompt (CrewAI pattern)
- **`WebFetchTool`**: fetches URL, returns cleaned markdown; optional `raw=True` for full HTML
- **`WebSearchTool`**: returns `[{title, url, snippet}]`; agent decides which URLs to fetch with WebFetchTool
- **tasks.md**: read/write by the agent on each REPL iteration (living state machine)
- **Done signal**: `final_answer()` tool — agent calls it when done; loop exits (smolagents pattern)
- **Validation**: simple retry on failure — inject error back to LLM, retry up to `max_retries`

### Decisions deferred to v2+
- `needs_approval=True` per tool (OpenAI SDK pattern) — can be added later
- Parallel guardrails — overkill for v1; simple retry is enough
- Multi-agent coordination / sub-agent fan-out
- Cost/token budget enforcement
- Forkable checkpoints

## Ranked Ideas

### 1. Self-Modifying tasks.md as Living State Machine
**Description:** Agent reads AND writes `tasks.md` on each REPL iteration — checking off completed steps (markdown checkboxes), appending discovered sub-tasks, annotating failures inline.
**Rationale:** Makes tasks.md an audit trail, resume checkpoint, and human-control surface simultaneously. Humans can intervene mid-run by editing the file. No other framework does this — genuine differentiator.
**Downsides:** File locking needed for concurrent access; must define a checkbox/status syntax convention.
**Confidence:** 92%
**Complexity:** Low
**Status:** Unexplored

### 2. Auto-Tool Registration from Python Type Hints
**Description:** Four ways to add tools: (a) pass a directory path in `tools=[]` for auto-discovery of `.py` files, (b) pass raw callables directly, (c) use `@agent.tool` decorator for inline definitions with typed deps injection, (d) use built-in tool objects (`FileSystemTool`, `ShellTool`, `WebFetchTool`, `WebSearchTool`). All use docstrings + type hints for schema generation.
**Rationale:** Every major framework converges on this pattern. Removes all tool registration boilerplate. The `@agent.tool` decorator with `RunContext[Deps]` (from PydanticAI) adds clean dependency injection.
**Downsides:** Brittle with complex types; requires good docstrings.
**Confidence:** 90%
**Complexity:** Low
**Status:** Unexplored

### 3. `final_answer` Tool as Done Signal
**Description:** Instead of requiring structured JSON every iteration, the agent calls `final_answer(result)` as a regular tool when it's done. The REPL loop exits when this tool is invoked. Combined with `max_steps` as a safety cap.
**Rationale:** Simpler than structured done-protocol (no forced JSON every turn). smolagents proves this works. Natural for the LLM — it's just another tool call.
**Downsides:** Agent might forget to call it; `max_steps` safety cap needed.
**Confidence:** 87%
**Complexity:** Low
**Status:** Unexplored

### 4. Error Recovery with Retry
**Description:** On tool failure: inject the error as a message to the LLM and let it decide what to do (retry, try alternative, skip). On LLM output validation failure: retry up to `max_retries` with the validation error as feedback. PydanticAI validates this pattern works well — LLMs self-correct when given their own error.
**Rationale:** Table-stakes for unattended operation. LLM self-correction via error feedback is proven effective.
**Downsides:** Retry logic can mask bugs.
**Confidence:** 85%
**Complexity:** Low
**Status:** Unexplored

### 5. Permissions via Tool Objects
**Description:** Directory access and shell access are tool objects with built-in constraints. No separate permission API. The application controls access by choosing which tools to pass and how to configure them.
**Rationale:** Consistent with JustAI's existing tool model. Every framework lacks a good permission story — this is simple and explicit.
**Downsides:** No implicit safety net if tools passed without constraints.
**Confidence:** 82%
**Complexity:** Low
**Status:** Unexplored

### 6. Typed Dependency Injection (`deps`)
**Description:** Agent accepts a `deps_type` (dataclass or Pydantic model). Dependencies are passed at `run()` time and available in tools and dynamic instructions via `ctx.deps`. No globals, trivially mockable for testing.
**Rationale:** Essential for use from applications. PydanticAI proves this pattern is both clean and popular. Database connections, API clients, user sessions — all flow through deps.
**Downsides:** Adds a type parameter to the Agent class; slight learning curve.
**Confidence:** 85%
**Complexity:** Low
**Status:** Unexplored

### 7. Dynamic Instructions via `@agent.instructions`
**Description:** Alongside static `.md` skill files, allow callable instruction fragments that run at the start of each agent run and compose into the system prompt. For injecting live context (current user, codebase state, database schema).
**Rationale:** Static `.md` skills go stale. Dynamic instructions keep context current. PydanticAI pattern — multiple decorators compose naturally.
**Downsides:** Dynamic instructions add a function call at startup; must be fast.
**Confidence:** 80%
**Complexity:** Low
**Status:** Unexplored

### 8. `role` / `goal` Identity Config
**Description:** Simple string parameters auto-injected into the system prompt. Quick way to set agent identity without writing a full skill file.
**Rationale:** CrewAI's most praised feature. Maps naturally to how people think about agents. Complements detailed `.md` skills.
**Downsides:** Minimal — just string interpolation into the system prompt.
**Confidence:** 85%
**Complexity:** Low
**Status:** Unexplored

### 9. Built-in Web Tools
**Description:** `WebFetchTool()` fetches a URL and returns cleaned markdown (optional `raw=True` for full HTML). `WebSearchTool()` returns `[{title, url, snippet}]` — agent decides which URLs to fetch.
**Rationale:** Web access is a near-universal agent requirement. Consistent interface with other built-in tools.
**Downsides:** Search API keys needed for reliable search; JS-heavy pages may need playwright.
**Confidence:** 85%
**Complexity:** Low-Medium
**Status:** Unexplored

### 10. Audit Trail for Filesystem/Shell Actions
**Description:** Every file mutation and shell command appended to a structured append-only log (path, command, timestamp, task context, outcome).
**Rationale:** Essential for trust when granting write/shell access. No framework does this well — real gap.
**Downsides:** Log can grow large; must decide storage location.
**Confidence:** 78%
**Complexity:** Low
**Status:** Unexplored

## Research Summary

Researched 7 frameworks: LangChain/LangGraph, LlamaIndex, AutoGen, CrewAI, smolagents, OpenAI Agents SDK, PydanticAI.

### Patterns adopted
| Pattern | Source | Applied to |
|---|---|---|
| `@agent.tool` with `RunContext[Deps]` | PydanticAI | Tool registration + dependency injection |
| `@agent.instructions` composable fragments | PydanticAI | Dynamic skill loading |
| `final_answer()` as done signal | smolagents | Agent loop termination |
| `role` / `goal` identity config | CrewAI | System prompt construction |
| Auto-retry on validation failure | PydanticAI | Error recovery |
| Docstring + type hints = tool schema | All frameworks | Tool registration |

### Patterns rejected for v1
| Pattern | Source | Reason |
|---|---|---|
| `needs_approval=True` per tool | OpenAI SDK | Can be added later; not needed for v1 |
| Parallel guardrails | OpenAI SDK | Overkill; simple retry is enough |
| Graph-based execution | LangGraph | Over-engineering for single-agent |
| Event-driven workflows | LlamaIndex | Adds complexity without clear benefit |
| AST interpreter sandbox | smolagents | Interesting but not our execution model |
| Code generation as tool-calling | smolagents | We use standard tool-calling, not code gen |
| Actor model message-passing | AutoGen | Multi-agent is v2+ |

### Key insight
Every framework over-engineers the common case. Our differentiators are:
1. `.md` skill files with cross-references — nobody else has this
2. Self-modifying `tasks.md` as state machine — unique control surface
3. Permissions built into tool objects — cleaner than any framework's approach

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | Skill Dependency Graph (as DAG engine) | A simple recursive `.md` include resolver handles the spec |
| 2 | Forkable Checkpoint-Based Lifecycle | v2+; solve correctness first |
| 3 | Tools via Skill Frontmatter | Duplicates Auto-Tool Registration via a more fragile mechanism |
| 4 | Provider-Agnostic Model Routing per Step | Premature optimization |
| 5 | Parallel Sub-Agent Fan-Out | v2+; get single-agent correctness first |
| 6 | LLM-Generated System Prompt | Good .md authoring solves this |
| 7 | Skills as Executable Notebooks | Security surface too large |
| 8 | Cost/Token Budget Enforcement | v2+; solve correctness first |
| 9 | Agent-Generated Task Graph | tasks.md is the explicit control surface |
| 10 | Skills Auto-Synthesized from Traces | LLMs are unreliable narrators |
| 11 | Automatic Verbosity Calibration | Simple flag achieves 90% of value |
| 12 | Structured Working Memory (typed slots) | tasks.md already serves this role |
| 13 | Episodic Memory with Semantic Retrieval | Entire product feature, not an agent primitive |
| 14 | Parallel Guardrails | Overkill for v1; simple retry is enough |
| 15 | `needs_approval` per tool | Can be added later |

## Session Log
- 2026-04-07: Initial ideation — ~30 candidates generated across 4 frames, 7 survivors + 1 added from discussion (WebTools). Key design decisions: permissions via tool objects (not separate API), mixed tool list (objects/dirs/callables), WebFetchTool returns markdown, WebSearchTool returns snippets only.
- 2026-04-07: Research phase — studied 7 frameworks (LangChain, LlamaIndex, AutoGen, CrewAI, smolagents, OpenAI Agents SDK, PydanticAI). Adopted: `@agent.tool` + typed deps (PydanticAI), `final_answer()` done signal (smolagents), `role/goal` config (CrewAI), dynamic `@agent.instructions` (PydanticAI). Rejected for v1: `needs_approval`, parallel guardrails. Refined to 10 ranked ideas.
