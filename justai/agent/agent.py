"""Agent class for autonomous agent execution with streaming events."""
import inspect
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

from justai.agent.skills import load_skills
from justai.model.model import Model
from justai.models.basemodel import (
    StreamChunk, ToolCallRequest,
    RatelimitException, AuthorizationException,
)

logger = logging.getLogger(__name__)


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


def _build_tool_schema(func: Callable) -> dict:
    """Build a tool schema from a callable's type hints and docstring."""
    sig = inspect.signature(func)
    type_map = {str: 'string', int: 'integer', float: 'number', bool: 'boolean', list: 'array', dict: 'object'}

    properties = {}
    required = []
    for name, param in sig.parameters.items():
        if name in ('self', 'ctx'):
            continue
        annotation = param.annotation
        json_type = type_map.get(annotation, 'string') if annotation != inspect.Parameter.empty else 'string'
        properties[name] = {'type': json_type, 'description': name}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        'name': func.__name__,
        'description': (func.__doc__ or func.__name__).strip(),
        'parameters': properties,
        'required': required,
    }


class Agent:
    def __init__(
        self,
        model: str | Model,
        role: str = '',
        goal: str = '',
        skills_dir: str | None = None,
        tools: list | None = None,
        max_retries: int = 3,
        max_iterations: int = 50,
        verbose: bool = True,
        **model_kwargs,
    ):
        if isinstance(model, str):
            self.model = Model(model, **model_kwargs)
        else:
            self.model = model

        self.role = role
        self.goal = goal
        self.skills_dir = skills_dir
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Tool registry: name -> (callable, schema, needs_ctx)
        self._tools: dict[str, tuple[Callable, dict, bool]] = {}
        self._instruction_fns: list[Callable] = []
        self._audit: list[AuditEntry] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._answer: str = ''

        # Register built-in final_answer tool
        self._register_tool(self._final_answer, needs_ctx=False, name='final_answer')

        # Register tools from tool objects and raw callables
        for t in (tools or []):
            if hasattr(t, 'get_tools'):
                for name, desc, params, func in t.get_tools():
                    schema = {
                        'name': name,
                        'description': desc,
                        'parameters': {k: {'type': _python_type_to_json(v)} for k, v in params.items()},
                        'required': list(params.keys()),
                    }
                    self._tools[name] = (func, schema, False)
            elif callable(t):
                self._register_tool(t, needs_ctx=False)

    def _final_answer(self, answer: str) -> str:
        """Submit the final answer and stop the agent loop."""
        self._answer = answer
        return 'Final answer submitted.'

    def _register_tool(self, func: Callable, needs_ctx: bool = False, name: str | None = None):
        """Register a callable as a tool."""
        schema = _build_tool_schema(func)
        tool_name = name or func.__name__
        schema['name'] = tool_name
        self._tools[tool_name] = (func, schema, needs_ctx)

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
        """Build tool specifications for the provider."""
        specs = []
        for name, (func, schema, needs_ctx) in self._tools.items():
            # Build in provider-agnostic format; Model.stream() handles conversion
            input_schema = {
                'type': 'object',
                'properties': schema['parameters'],
                'required': schema.get('required', []),
            }
            specs.append({
                'name': name,
                'description': schema['description'],
                'input_schema': input_schema,
            })
        return specs

    def _execute_tool(self, tc: ToolCallRequest, ctx: AgentContext) -> tuple[str, bool]:
        """Execute a tool call, return (result_str, success)."""
        if tc.name not in self._tools:
            available = ', '.join(self._tools.keys())
            return f'Error: tool "{tc.name}" not found. Available tools: {available}', False

        func, schema, needs_ctx = self._tools[tc.name]
        start = time.time()
        try:
            if needs_ctx:
                result = func(ctx, **tc.arguments)
            else:
                result = func(**tc.arguments)
            result_str = str(result) if not isinstance(result, str) else result
            success = True
        except Exception as e:
            result_str = f'Error ({type(e).__name__}): {e}'
            success = False
            logger.warning(f'Tool {tc.name} failed: {e}')

        duration_ms = int((time.time() - start) * 1000)
        self._audit.append(AuditEntry(
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
            tool_name=tc.name,
            arguments=tc.arguments,
            result=result_str[:500],
            duration_ms=duration_ms,
            success=success,
        ))
        return result_str, success

    def _read_tasks(self, tasks_file: str) -> str:
        """Read tasks file content."""
        try:
            return open(tasks_file).read()
        except FileNotFoundError:
            return ''

    def _write_tasks(self, tasks_file: str, content: str):
        """Atomically write tasks file."""
        dir_name = os.path.dirname(os.path.abspath(tasks_file))
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(content)
            os.replace(tmp_path, tasks_file)
        except Exception:
            os.unlink(tmp_path)
            raise

    def _build_result(self, tasks_content: str, iterations: int) -> AgentResult:
        """Build the final AgentResult."""
        return AgentResult(
            answer=self._answer or '(no final answer submitted)',
            audit=list(self._audit),
            tasks=tasks_content,
            tokens=(self._total_input_tokens, self._total_output_tokens),
            iterations=iterations,
        )

    async def run(self, tasks_file: str, deps: Any = None) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent loop, yielding events as it progresses."""
        ctx = AgentContext(deps=deps, agent=self)
        self._audit = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._answer = ''

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
                except RatelimitException:
                    retry_count += 1
                    if retry_count > self.max_retries:
                        yield AgentEvent(type='error', message='Rate limit exceeded, max retries reached')
                        yield AgentEvent(type='done', result=self._build_result(tasks_content, iteration + 1))
                        return
                    yield AgentEvent(type='status', message=f'Rate limited, retrying ({retry_count}/{self.max_retries})...')
                    import asyncio
                    await asyncio.sleep(2 ** retry_count)
                except AuthorizationException as e:
                    yield AgentEvent(type='error', message=f'Authorization error: {e}')
                    yield AgentEvent(type='done', result=self._build_result(tasks_content, iteration + 1))
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
                    result_str, success = self._execute_tool(tc, ctx)
                    yield AgentEvent(
                        type='tool_call', name=tc.name,
                        arguments=tc.arguments, tool_result=result_str
                    )
                    tool_results.append(
                        self.model.model.format_tool_result(tc.id, tc.name, result_str)
                    )

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


def _python_type_to_json(t: type) -> str:
    """Map Python types to JSON Schema types."""
    mapping = {str: 'string', int: 'integer', float: 'number', bool: 'boolean', list: 'array', dict: 'object'}
    return mapping.get(t, 'string')
