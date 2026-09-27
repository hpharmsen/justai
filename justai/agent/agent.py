"""Agent class for autonomous agent execution with streaming events."""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from justai.agent.skills import load_skills
from justai.model.model import Model, _to_pydantic
from justai.models.basemodel import (
    DEFAULT_AGENT_RETRIES,
    AuthorizationException,
    BadRequestException,
    ConnectionException,
    GeneralException,
    ModelOverloadException,
    RatelimitException,
    RefusalException,
    TimeoutException,
    ToolCallRequest,
)
from justai.tools.validation import TOOL_FEEDBACK, RepairBudget

logger = logging.getLogger(__name__)

# Provider failures that end the run with an 'error' event plus 'done'. Anything else is a
# bug and raises. GeneralException also covers TruncatedResponseException.
STOP_EXCEPTIONS = (
    AuthorizationException,
    BadRequestException,
    GeneralException,
    ModelOverloadException,
    RefusalException,
    TimeoutException,
)


@dataclass
class AgentContext:
    """Context passed to tools and instructions."""

    deps: Any = None
    agent: Any = None


@dataclass
class AuditEntry:
    """A single tool execution record."""

    timestamp: str
    tool_name: str
    arguments: dict
    result: str
    duration_ms: int
    success: bool


@dataclass
class AgentResult:
    """Final result of an agent run."""

    answer: str
    audit: list[AuditEntry] = field(default_factory=list)
    tasks: str = ''
    tokens: tuple[int, int] = (0, 0)
    iterations: int = 0
    error: str | None = None  # Set when a provider failure or exhausted validation retries stopped the run


@dataclass
class AgentEvent:
    """An event yielded during agent execution."""

    type: str  # 'status' | 'response' | 'tool_call' | 'task_update' | 'error' | 'done'
    message: str | None = None
    content: str | None = None
    name: str | None = None
    arguments: dict | None = None
    tool_result: str | None = None
    result: AgentResult | None = None


def _build_args_model(name: str, func: Callable, types: dict[str, Any] | None = None) -> type[BaseModel]:
    """Build a strict args model from func's signature; `types` overrides the annotations (get_tools objects)."""
    kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    all_params = inspect.signature(func).parameters
    params = {k: p for k, p in all_params.items() if k not in ('self', 'ctx') and p.kind not in kinds}
    for k in params:
        assert not k.startswith('_') and k != 'model_config', f'Tool {name}: parameter name {k!r} is reserved'
    if types is None:
        hints = get_type_hints(func, include_extras=True)  # resolves `from __future__ import annotations` strings
        types = {k: hints.get(k, Any) for k in params}
    empty = inspect.Parameter.empty
    fields = {
        k: (t, params[k].default if k in params and params[k].default is not empty else ...) for k, t in types.items()
    }
    extra = 'allow' if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in all_params.values()) else 'forbid'
    return create_model(f'{name}_args', __config__=ConfigDict(extra=extra), **fields)


class Agent:
    def __init__(
        self,
        model: str | Model,
        role: str = '',
        goal: str = '',
        skills_dir: str | None = None,
        tools: list | None = None,
        max_retries: int | None = None,
        max_iterations: int = 50,
        verbose: bool = True,
        validation_retries: int = 2,
        **model_kwargs,
    ):
        if isinstance(model, str):
            self.model = Model(model, **model_kwargs)
        else:
            self.model = model

        self.role = role
        self.goal = goal
        self.skills_dir = skills_dir
        # One knob for both layers: Model(..., max_retries=0) has to switch off the
        # agent's retries too, otherwise a caller under a deadline still gets three.
        if max_retries is None:
            max_retries = self.model.model.model_params.get('max_retries', DEFAULT_AGENT_RETRIES)
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.validation_retries = validation_retries

        # Tool registry: name -> (callable, description, needs_ctx, args_model)
        self._tools: dict[str, tuple[Callable, str, bool, type[BaseModel]]] = {}
        self._instruction_fns: list[Callable] = []
        self._audit: list[AuditEntry] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._answer: str = ''
        self._error: str | None = None

        # Register built-in final_answer tool
        self._register_tool(self._final_answer, needs_ctx=False, name='final_answer')

        # Register tools from tool objects and raw callables
        for t in tools or []:
            if hasattr(t, 'get_tools'):
                for name, desc, params, func in t.get_tools():
                    self._tools[name] = (func, desc, False, _build_args_model(name, func, params))
            elif callable(t):
                self._register_tool(t, needs_ctx=False)

    def _final_answer(self, answer: str) -> str:
        """Submit the final answer and stop the agent loop."""
        self._answer = answer
        return 'Final answer submitted.'

    def _register_tool(self, func: Callable, needs_ctx: bool = False, name: str | None = None):
        """Register a callable as a tool."""
        tool_name = name or func.__name__
        description = (func.__doc__ or func.__name__).strip()
        self._tools[tool_name] = (func, description, needs_ctx, _build_args_model(tool_name, func))

    def tool(self, func: Callable) -> Callable:
        """Decorator to register a tool with context injection."""
        self._register_tool(func, needs_ctx=True)
        return func

    def instructions(self, func: Callable) -> Callable:
        """Decorator to register a dynamic instructions function."""
        self._instruction_fns.append(func)
        return func

    def _build_system_prompt(self, ctx: AgentContext) -> str:
        """Compose system prompt from role, goal, skills, and instructions."""
        parts = []
        if self.role:
            parts.append(f'You are a {self.role}.')
        if self.goal:
            parts.append(f'Your goal: {self.goal}')

        # Skills
        if self.skills_dir:
            skills_text = load_skills(self.skills_dir)
            if skills_text:
                parts.append(skills_text)

        # Dynamic instructions (evaluated once per run)
        for fn in self._instruction_fns:
            result = fn(ctx)
            if result:
                parts.append(result)

        # Tool usage instructions
        tool_names = list(self._tools.keys())
        parts.append(
            'You have access to tools. Use them to accomplish your tasks. '
            f'Available tools: {", ".join(tool_names)}. '
            'When all tasks are complete, call final_answer with your summary.'
        )

        return '\n\n'.join(parts)

    def _build_tool_specs(self) -> list[dict]:
        """Build provider-agnostic tool specs; the args model's JSON schema is the single source of truth."""
        return [
            {'name': name, 'description': desc, 'input_schema': args_model.model_json_schema()}
            for name, (_, desc, _, args_model) in self._tools.items()
        ]

    def _check_args(self, tc: ToolCallRequest, budgets: dict[str, RepairBudget]) -> tuple[dict | None, str | None]:
        """Validate a call's arguments: (kwargs, None), (None, feedback), or (None, None) for an unknown tool."""
        if tc.name not in self._tools:
            return None, None
        if tc.name not in budgets:
            args_model = self._tools[tc.name][3]
            budgets[tc.name] = RepairBudget(
                lambda raw: _to_pydantic(raw, args_model), self.validation_retries, tc.name, TOOL_FEEDBACK
            )
        instance, feedback = budgets[tc.name].check(tc.raw_arguments if tc.raw_arguments is not None else tc.arguments)
        return (None, feedback) if feedback else (dict(instance), None)

    def _execute_tool(self, tc: ToolCallRequest, ctx: AgentContext, kwargs: dict | None = None) -> tuple[str, bool]:
        """Execute a tool call with validated kwargs (default: the raw arguments), return (result_str, success)."""
        if tc.name not in self._tools:
            available = ', '.join(self._tools.keys())
            return f'Error: tool "{tc.name}" not found. Available tools: {available}', False

        func, _, needs_ctx, _ = self._tools[tc.name]
        kwargs = tc.arguments if kwargs is None else kwargs
        start = time.time()
        try:
            result = func(ctx, **kwargs) if needs_ctx else func(**kwargs)
            result_str = str(result) if not isinstance(result, str) else result
            success = True
        except Exception as e:
            result_str = f'Error ({type(e).__name__}): {e}'
            success = False
            logger.warning(f'Tool {tc.name} failed: {e}')

        duration_ms = int((time.time() - start) * 1000)
        self._audit.append(
            AuditEntry(
                timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
                tool_name=tc.name,
                arguments=tc.arguments,
                result=result_str[:500],
                duration_ms=duration_ms,
                success=success,
            )
        )
        return result_str, success

    def _read_tasks(self, tasks_file: str) -> str:
        """Read tasks file content."""
        try:
            return open(tasks_file).read()
        except FileNotFoundError:
            return ''

    def _build_result(self, tasks_content: str, iterations: int) -> AgentResult:
        """Build the final AgentResult."""
        return AgentResult(
            answer=self._answer or '(no final answer submitted)',
            audit=list(self._audit),
            tasks=tasks_content,
            tokens=(self._total_input_tokens, self._total_output_tokens),
            iterations=iterations,
            error=self._error,
        )

    def _stop(self, message: str, tasks_content: str, iterations: int) -> list[AgentEvent]:
        """End the run on a provider failure or exhausted validation retries: log, then 'error' plus 'done'."""
        logger.error(f'Agent stopped: {message}')
        self._error = message
        return [
            AgentEvent(type='error', message=message),
            AgentEvent(type='done', result=self._build_result(tasks_content, iterations)),
        ]

    async def run(self, tasks_file: str, deps: Any = None) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent loop, yielding events as it progresses."""
        ctx = AgentContext(deps=deps, agent=self)
        self._audit = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._answer = ''
        self._error = None

        # Build system prompt and tool specs
        system_prompt = self._build_system_prompt(ctx)
        tool_specs = self._build_tool_specs()

        # Read tasks
        tasks_content = self._read_tasks(tasks_file)

        # Initialize messages
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': tasks_content or 'No tasks file provided. Ask the user what to do.'},
        ]

        yield AgentEvent(type='status', message='Agent started')
        budgets: dict[str, RepairBudget] = {}  # per tool name, for this run

        for iteration in range(self.max_iterations):
            response_text = ''
            tool_calls: list[ToolCallRequest] = []

            # Stream from model
            retry_count = 0
            while True:
                try:
                    async for chunk in self.model.model.stream(messages, tool_specs):
                        if chunk.type == 'text' and chunk.content:
                            response_text += chunk.content
                            if self.verbose:
                                yield AgentEvent(type='response', content=chunk.content)
                        elif chunk.type == 'tool_calls':
                            tool_calls = chunk.tool_calls
                        elif chunk.type == 'done':
                            if chunk.input_tokens:
                                self._total_input_tokens += chunk.input_tokens
                            if chunk.output_tokens:
                                self._total_output_tokens += chunk.output_tokens
                    break  # Success
                except (RatelimitException, ConnectionException) as e:
                    # A failed stream is only safe to replay while nothing has left this
                    # step yet. Once text has streamed to the caller or a tool call is
                    # pending, a retry would duplicate output or re-run side effects.
                    kind = 'Rate limit' if isinstance(e, RatelimitException) else 'Connection lost'
                    retry_count += 1
                    if response_text or tool_calls:
                        message = f'{kind} mid-response, not retried: {e}'
                    elif retry_count > self.max_retries:
                        message = f'{kind}, max retries reached: {e}'
                    else:
                        yield AgentEvent(
                            type='status', message=f'{kind}, retrying ({retry_count}/{self.max_retries})...'
                        )
                        await asyncio.sleep(2**retry_count)
                        continue
                    for event in self._stop(message, tasks_content, iteration + 1):
                        yield event
                    return
                except STOP_EXCEPTIONS as e:
                    for event in self._stop(f'{type(e).__name__}: {e}', tasks_content, iteration + 1):
                        yield event
                    return

            # Build assistant message in provider-specific format
            assistant_msgs = self.model.model.format_assistant_message(
                response_text, tool_calls if tool_calls else None
            )
            messages.extend(assistant_msgs)

            # Execute tool calls
            if tool_calls:
                tool_results = []
                for tc in tool_calls:
                    try:
                        kwargs, feedback = self._check_args(tc, budgets)
                    except Exception as e:  # Spent budget, or user validator code raising: stop, as _execute_tool does
                        for event in self._stop(f'{type(e).__name__}: {e}', tasks_content, iteration + 1):
                            yield event
                        return
                    if feedback:
                        n, total = budgets[tc.name].failures, self.validation_retries
                        yield AgentEvent(type='status', message=f'Invalid arguments for {tc.name}, retry {n}/{total}')
                        tool_results.append(self.model.model.format_tool_result(tc.id, tc.name, feedback))
                        continue
                    result_str, success = self._execute_tool(tc, ctx, kwargs)
                    yield AgentEvent(type='tool_call', name=tc.name, arguments=tc.arguments, tool_result=result_str)
                    tool_results.append(self.model.model.format_tool_result(tc.id, tc.name, result_str))

                # Add tool results to messages
                # For Anthropic: each result is a user message with tool_result content
                # For OpenAI: each result is a function_call_output item
                # The format_tool_result already handles provider differences
                for tr in tool_results:
                    messages.append(tr)

                # Check if final_answer was called
                if self._answer:
                    yield AgentEvent(type='done', result=self._build_result(tasks_content, iteration + 1))
                    return

            else:
                # No tool calls — check if this is a final response
                # If the LLM responded without tool calls, treat it as done
                if not self._answer:
                    self._answer = response_text
                yield AgentEvent(type='done', result=self._build_result(tasks_content, iteration + 1))
                return

            # Update tasks file
            if tasks_file and tasks_content:
                yield AgentEvent(type='task_update', message=f'Iteration {iteration + 1} complete')

        # Max iterations reached
        yield AgentEvent(type='status', message=f'Max iterations ({self.max_iterations}) reached')
        yield AgentEvent(type='done', result=self._build_result(tasks_content, self.max_iterations))

    async def run_until_done(self, tasks_file: str, deps: Any = None) -> AgentResult:
        """Run the agent and return the final result."""
        result = None
        async for event in self.run(tasks_file, deps):
            if event.type == 'done':
                result = event.result
        assert result is not None, 'Agent did not produce a result'
        return result
