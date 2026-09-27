"""Tests for tool argument validation with repair retries in the Agent (no network)."""

import asyncio

from pydantic import BaseModel

from justai import Agent, AgentEvent
from justai.models.basemodel import StreamChunk, ToolCallRequest

FINAL = ToolCallRequest('final', 'final_answer', {'answer': 'ok'})


class InvoiceArgs(BaseModel):
    number: str
    amount: int


def tc(id_: str, name: str, arguments: dict, raw: str | None = None) -> ToolCallRequest:
    """Shorthand for a tool call request."""
    return ToolCallRequest(id_, name, arguments, raw_arguments=raw)


def make_agent(turns: list[list[ToolCallRequest]], **kwargs) -> tuple[Agent, list[list]]:
    """Agent whose stream yields one scripted list of tool calls per call; returns (agent, message snapshots)."""
    agent = Agent('claude-sonnet-4-6', ANTHROPIC_API_KEY='k', verbose=False, **kwargs)
    snapshots = []

    async def stream(messages, tools):
        snapshots.append(list(messages))
        yield StreamChunk(type='tool_calls', tool_calls=turns[len(snapshots) - 1])
        yield StreamChunk(type='done')

    agent.model.model.stream = stream
    return agent, snapshots


def run(agent: Agent, tmp_path) -> list[AgentEvent]:
    """Run the agent on a tasks file and collect all events."""
    tasks = tmp_path / 'tasks.md'
    tasks.write_text('- do it')

    async def collect():
        return [e async for e in agent.run(str(tasks))]

    return asyncio.run(collect())


def tool_results(messages: list) -> dict[str, str]:
    """Map tool_use_id to tool result content across the messages."""
    return {
        block['tool_use_id']: block['content']
        for m in messages
        if isinstance(m.get('content'), list)
        for block in m['content']
        if block.get('type') == 'tool_result'
    }


def recorder(agent: Agent, name: str = 'add'):
    """Register a tool `add(a: int, b: int = 0)` that records its calls."""
    calls = []

    def add(a: int, b: int = 0) -> str:
        """Add two numbers."""
        calls.append((a, b))
        return str(a + b)

    add.__name__ = name
    agent._register_tool(add)
    return calls


def test_valid_args_run_tool_once(tmp_path):
    agent, _ = make_agent([[tc('1', 'add', {'a': 1, 'b': 2})], [FINAL]])
    calls = recorder(agent)
    events = run(agent, tmp_path)
    assert calls == [(1, 2)]
    assert events[-1].type == 'done' and events[-1].result.error is None


def test_invalid_then_valid(tmp_path):
    agent, snaps = make_agent([[tc('1', 'add', {'a': 'x'})], [tc('2', 'add', {'a': 3})], [FINAL]])
    calls = recorder(agent)
    events = run(agent, tmp_path)
    feedback = tool_results(snaps[1])['1']
    assert 'Tool call `add`' in feedback and 'a:' in feedback
    assert calls == [(3, 0)]
    assert events[-1].result.error is None


def test_status_event_text(tmp_path):
    agent, _ = make_agent([[tc('1', 'add', {'a': 'x'})], [FINAL]])
    recorder(agent)
    events = run(agent, tmp_path)
    assert 'Invalid arguments for add, retry 1/2' in [e.message for e in events if e.type == 'status']


def test_coerced_values_reach_tool(tmp_path):
    agent, _ = make_agent([[tc('1', 'add', {'a': '12'})], [FINAL]])
    calls = recorder(agent)
    run(agent, tmp_path)
    assert calls == [(12, 0)]


def test_pydantic_param_arrives_as_instance(tmp_path):
    agent, _ = make_agent([[tc('1', 'book', {'args': {'number': 'F1', 'amount': '5'}})], [FINAL]])
    received = []

    def book(args: InvoiceArgs) -> str:
        """Book an invoice."""
        received.append(args)
        return 'booked'

    agent._register_tool(book)
    run(agent, tmp_path)
    assert received == [InvoiceArgs(number='F1', amount=5)]
    assert isinstance(received[0], InvoiceArgs)


def test_pydantic_param_schema_is_object(tmp_path):
    agent, _ = make_agent([])

    def book(args: InvoiceArgs) -> str:
        """Book an invoice."""
        return 'booked'

    agent._register_tool(book)
    prop = agent._tools['book'][1]['parameters']['args']
    assert prop['type'] == 'object' and 'amount' in prop['properties']


def test_extra_argument_rejected(tmp_path):
    agent, snaps = make_agent([[tc('1', 'add', {'a': 1, 'c': 2})], [FINAL]])
    calls = recorder(agent)
    events = run(agent, tmp_path)
    assert calls == []
    assert 'c:' in tool_results(snaps[1])['1']
    assert not [e for e in events if e.type == 'tool_call' and e.name == 'add']


def test_always_invalid_never_runs(tmp_path):
    agent, snaps = make_agent([[tc(str(i), 'add', {'a': 'x'})] for i in range(10)])
    calls = recorder(agent)
    events = run(agent, tmp_path)
    assert calls == []
    assert [e.type for e in events[-2:]] == ['error', 'done']
    assert events[-1].result.error == events[-2].message
    assert 'add' in events[-1].result.error
    assert len(snaps) == 1 + agent.validation_retries


def test_zero_retries_stops_at_first_invalid(tmp_path):
    agent, snaps = make_agent([[tc('1', 'add', {'a': 'x'})], [FINAL]], validation_retries=0)
    calls = recorder(agent)
    events = run(agent, tmp_path)
    assert calls == [] and len(snaps) == 1
    assert [e.type for e in events[-2:]] == ['error', 'done']


def test_counters_per_tool(tmp_path):
    agent, snaps = make_agent(
        [
            [tc('1', 'add', {'a': 'x'})],
            [tc('2', 'mul', {'a': 2})],
            [tc('3', 'add', {'a': 'x'})],
            [tc('4', 'add', {'a': 'x'})],
            [FINAL],
        ]
    )
    add_calls, mul_calls = recorder(agent), recorder(agent, 'mul')
    events = run(agent, tmp_path)
    assert mul_calls == [(2, 0)] and add_calls == []
    # A's counter kept running across B's success: third failure of add exhausts budget 2
    assert [e.type for e in events[-2:]] == ['error', 'done']
    assert len(snaps) == 4


def test_counter_resets_after_success(tmp_path):
    agent, _ = make_agent(
        [
            [tc('1', 'add', {'a': 'x'})],
            [tc('2', 'add', {'a': 'x'})],
            [tc('3', 'add', {'a': 1})],
            [tc('4', 'add', {'a': 'x'})],
            [tc('5', 'add', {'a': 'x'})],
            [FINAL],
        ]
    )
    calls = recorder(agent)
    events = run(agent, tmp_path)
    assert calls == [(1, 0)]
    assert events[-1].type == 'done' and events[-1].result.error is None
    assert events[-1].result.answer == 'ok'


def test_two_invalid_calls_same_turn(tmp_path):
    agent, snaps = make_agent([[tc('a1', 'add', {'a': 'x'}), tc('m1', 'mul', {'b': 1})], [FINAL]])
    recorder(agent), recorder(agent, 'mul')
    run(agent, tmp_path)
    results = tool_results(snaps[1])
    assert 'Tool call `add`' in results['a1'] and 'a:' in results['a1']
    assert 'Tool call `mul`' in results['m1']


def test_malformed_json_args_repairable(tmp_path):
    agent, snaps = make_agent([[tc('1', 'add', {}, raw='{"a": ')], [tc('2', 'add', {'a': 4})], [FINAL]])
    calls = recorder(agent)
    events = run(agent, tmp_path)
    assert 'Invalid JSON' in tool_results(snaps[1])['1']
    assert calls == [(4, 0)] and events[-1].result.error is None


def test_ctx_excluded_from_args_model(tmp_path):
    agent, _ = make_agent([[tc('1', 'lookup', {'query': 'q'})], [FINAL]])
    seen = []

    @agent.tool
    def lookup(ctx, query: str) -> str:
        """Look something up."""
        seen.append((ctx.deps, query))
        return 'found'

    assert 'ctx' not in agent._tools['lookup'][3].model_fields
    run(agent, tmp_path)
    assert seen == [(None, 'q')]


class GreetTool:
    """Small tool object exposing get_tools()."""

    def __init__(self):
        self.calls = []

    def greet(self, name: str, loud: bool = False) -> str:
        self.calls.append((name, loud))
        return f'hi {name}'

    def get_tools(self) -> list[tuple]:
        return [('greet', 'Greet someone.', {'name': str, 'loud': bool}, self.greet)]


def test_get_tools_object_validated(tmp_path):
    tool = GreetTool()
    agent, snaps = make_agent(
        [[tc('1', 'greet', {'name': 5})], [tc('2', 'greet', {'name': 'Bo'})], [FINAL]], tools=[tool]
    )
    run(agent, tmp_path)
    assert 'Tool call `greet`' in tool_results(snaps[1])['1']
    assert tool.calls == [('Bo', False)]


def test_no_audit_entry_for_rejected_call(tmp_path):
    agent, _ = make_agent([[tc('1', 'add', {'a': 'x'})], [tc('2', 'add', {'a': 1})], [FINAL]])
    recorder(agent)
    events = run(agent, tmp_path)
    audit = events[-1].result.audit
    assert [(a.tool_name, a.arguments) for a in audit] == [('add', {'a': 1}), ('final_answer', {'answer': 'ok'})]
