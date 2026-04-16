"""Tests for the Agent class, built-in tools, and skills loader.

Usage:
    python tests/test_agent.py
"""
import asyncio
import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from justai import Agent, AgentEvent, AgentResult, AgentContext, Model, FileSystemTool, ShellTool, WebFetchTool
from justai.agent.skills import load_skills
from justai.models.basemodel import ToolCallRequest, StreamChunk


# ──────────────────────────────────────────────
# Unit tests (no API calls)
# ──────────────────────────────────────────────

def test_filesystem_tool_read_write():
    """Test FileSystemTool read/write with path traversal prevention."""
    with tempfile.TemporaryDirectory() as tmpdir:
        read_dir = Path(tmpdir) / 'read'
        write_dir = Path(tmpdir) / 'write'
        read_dir.mkdir()
        write_dir.mkdir()

        # Write a test file
        (read_dir / 'test.txt').write_text('hello world')

        tool = FileSystemTool(read=[str(read_dir)], write=[str(write_dir)])

        # Read works
        assert tool.read_file(str(read_dir / 'test.txt')) == 'hello world'

        # Write works
        result = tool.write_file(str(write_dir / 'output.txt'), 'written')
        assert 'Written' in result
        assert (write_dir / 'output.txt').read_text() == 'written'

        # List directory works
        listing = tool.list_directory(str(read_dir))
        assert 'test.txt' in listing

        # Path traversal blocked
        try:
            tool.read_file(str(write_dir / 'output.txt'))
            assert False, 'Should have raised PermissionError'
        except PermissionError:
            pass

        try:
            tool.write_file(str(read_dir / 'hack.txt'), 'bad')
            assert False, 'Should have raised PermissionError'
        except PermissionError:
            pass

    print('  OK: filesystem tool read/write/path traversal')


def test_shell_tool():
    """Test ShellTool with allowlist and metacharacter rejection."""
    tool = ShellTool(allowlist=['echo', 'ls'])

    # Allowed command works
    result = tool.run_command('echo', ['hello'])
    assert 'hello' in result
    assert 'exit_code: 0' in result

    # Disallowed command blocked
    try:
        tool.run_command('rm', ['-rf', '/'])
        assert False, 'Should have raised PermissionError'
    except PermissionError as e:
        assert 'not allowed' in str(e)

    # Metacharacters blocked
    try:
        tool.run_command('echo', ['hello; rm -rf /'])
        assert False, 'Should have raised PermissionError'
    except PermissionError as e:
        assert 'metacharacters' in str(e)

    print('  OK: shell tool allowlist/metachar rejection')


def test_web_fetch_tool_ssrf():
    """Test WebFetchTool SSRF protection."""
    tool = WebFetchTool()

    # Private IP blocked
    try:
        tool.fetch_url('http://192.168.1.1/')
        assert False, 'Should have raised PermissionError'
    except PermissionError:
        pass

    # Non-http scheme blocked
    try:
        tool.fetch_url('file:///etc/passwd')
        assert False, 'Should have raised PermissionError'
    except PermissionError:
        pass

    print('  OK: web fetch SSRF protection')


def test_skills_loader():
    """Test skill loader reads and concatenates .md files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / 'a_skill.md').write_text('# Skill A\nDo A things')
        (Path(tmpdir) / 'b_skill.md').write_text('# Skill B\nDo B things')
        (Path(tmpdir) / 'not_a_skill.txt').write_text('ignored')

        result = load_skills(tmpdir)
        assert '# Skill A' in result
        assert '# Skill B' in result
        assert 'ignored' not in result
        # Alphabetical order
        assert result.index('Skill A') < result.index('Skill B')

    # Non-existent directory raises
    try:
        load_skills('/nonexistent/path')
        assert False, 'Should have raised FileNotFoundError'
    except FileNotFoundError:
        pass

    print('  OK: skills loader')


def test_tool_specs():
    """Test that get_tools() returns correct format."""
    fs = FileSystemTool(read=['/tmp'], write=['/tmp'])
    tools = fs.get_tools()
    assert len(tools) == 3  # read_file, list_directory, write_file
    for name, desc, params, func in tools:
        assert isinstance(name, str)
        assert isinstance(desc, str)
        assert isinstance(params, dict)
        assert callable(func)

    sh = ShellTool(allowlist=['ls'])
    tools = sh.get_tools()
    assert len(tools) == 1
    assert tools[0][0] == 'run_command'

    wf = WebFetchTool()
    tools = wf.get_tools()
    assert len(tools) == 1
    assert tools[0][0] == 'fetch_url'

    print('  OK: tool specs format')


def test_dataclasses():
    """Test ToolCallRequest and StreamChunk dataclasses."""
    tc = ToolCallRequest(id='123', name='test', arguments={'a': 1})
    assert tc.id == '123'
    assert tc.name == 'test'

    sc = StreamChunk(type='text', content='hello')
    assert sc.type == 'text'
    assert sc.content == 'hello'
    assert sc.tool_calls == []

    sc2 = StreamChunk(type='done', input_tokens=10, output_tokens=20)
    assert sc2.input_tokens == 10

    print('  OK: dataclasses')


def test_agent_init():
    """Test Agent initialization with various tool types."""
    agent = Agent(
        model='claude-sonnet-4-6',
        role='test agent',
        goal='test things',
        tools=[
            FileSystemTool(read=['/tmp']),
            ShellTool(allowlist=['ls']),
        ],
        max_iterations=5,
    )
    assert 'final_answer' in agent._tools
    assert 'read_file' in agent._tools
    assert 'run_command' in agent._tools

    # @agent.tool decorator
    @agent.tool
    def custom_tool(ctx, query: str) -> str:
        """Search something."""
        return f'result for {query}'

    assert 'custom_tool' in agent._tools
    assert agent._tools['custom_tool'][2] is True  # needs_ctx

    # @agent.instructions
    @agent.instructions
    def inject(ctx) -> str:
        return 'extra context'

    assert len(agent._instruction_fns) == 1

    print('  OK: agent initialization')


def test_agent_system_prompt():
    """Test system prompt composition."""
    agent = Agent(model='claude-sonnet-4-6', role='developer', goal='fix bugs')
    ctx = AgentContext(deps={'user': 'test'})
    prompt = agent._build_system_prompt(ctx)
    assert 'developer' in prompt
    assert 'fix bugs' in prompt
    assert 'final_answer' in prompt

    print('  OK: system prompt composition')


def test_agent_system_prompt_with_skills():
    """Test system prompt includes skills from directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / 'coding.md').write_text('Always write tests.')
        agent = Agent(model='claude-sonnet-4-6', role='coder', goal='write code', skills_dir=tmpdir)
        ctx = AgentContext()
        prompt = agent._build_system_prompt(ctx)
        assert 'Always write tests' in prompt
        assert 'coder' in prompt

    print('  OK: system prompt with skills')


def test_agent_system_prompt_with_instructions():
    """Test dynamic instructions are included in system prompt."""
    agent = Agent(model='claude-sonnet-4-6', role='helper', goal='help')

    @agent.instructions
    def add_context(ctx) -> str:
        return f'User is {ctx.deps["name"]}'

    ctx = AgentContext(deps={'name': 'Alice'})
    prompt = agent._build_system_prompt(ctx)
    assert 'User is Alice' in prompt

    print('  OK: system prompt with instructions')


def test_agent_final_answer_tool():
    """Test the built-in final_answer tool."""
    agent = Agent(model='claude-sonnet-4-6', role='test', goal='test')
    assert agent._answer == ''
    result = agent._final_answer(answer='done!')
    assert agent._answer == 'done!'
    assert 'submitted' in result.lower()

    print('  OK: final answer tool')


def test_agent_execute_unknown_tool():
    """Test executing a tool that doesn't exist."""
    agent = Agent(model='claude-sonnet-4-6', role='test', goal='test')
    ctx = AgentContext()
    tc = ToolCallRequest(id='1', name='nonexistent', arguments={})
    result_str, success = agent._execute_tool(tc, ctx)
    assert not success
    assert 'not found' in result_str

    print('  OK: unknown tool handling')


def test_agent_execute_tool_error():
    """Test tool execution that raises an exception."""
    agent = Agent(model='claude-sonnet-4-6', role='test', goal='test')

    def bad_tool() -> str:
        """A tool that always fails."""
        raise ValueError('something went wrong')

    agent._register_tool(bad_tool, needs_ctx=False)
    ctx = AgentContext()
    tc = ToolCallRequest(id='1', name='bad_tool', arguments={})
    result_str, success = agent._execute_tool(tc, ctx)
    assert not success
    assert 'ValueError' in result_str
    assert 'something went wrong' in result_str
    assert len(agent._audit) == 1
    assert not agent._audit[0].success

    print('  OK: tool execution error handling')


def test_agent_audit_trail():
    """Test that audit entries are recorded correctly."""
    agent = Agent(model='claude-sonnet-4-6', role='test', goal='test')
    ctx = AgentContext()

    # Execute final_answer to create an audit entry
    tc = ToolCallRequest(id='1', name='final_answer', arguments={'answer': 'test'})
    result_str, success = agent._execute_tool(tc, ctx)
    assert success
    assert len(agent._audit) == 1
    entry = agent._audit[0]
    assert entry.tool_name == 'final_answer'
    assert entry.success
    assert entry.duration_ms >= 0
    assert entry.timestamp

    print('  OK: audit trail')


def test_agent_tool_decorator_with_context():
    """Test that @agent.tool passes context correctly."""
    agent = Agent(model='claude-sonnet-4-6', role='test', goal='test')

    @agent.tool
    def greet(ctx, name: str) -> str:
        """Greet someone."""
        return f'hello {name} from {ctx.deps}'

    ctx = AgentContext(deps='world')
    tc = ToolCallRequest(id='1', name='greet', arguments={'name': 'Alice'})
    result_str, success = agent._execute_tool(tc, ctx)
    assert success
    assert result_str == 'hello Alice from world'

    print('  OK: tool decorator with context')


def test_agent_build_tool_specs():
    """Test tool spec generation format."""
    agent = Agent(model='claude-sonnet-4-6', role='test', goal='test',
                  tools=[FileSystemTool(read=['/tmp'])])
    specs = agent._build_tool_specs()
    names = {s['name'] for s in specs}
    assert 'final_answer' in names
    assert 'read_file' in names
    for spec in specs:
        assert 'name' in spec
        assert 'description' in spec
        assert 'input_schema' in spec
        assert spec['input_schema']['type'] == 'object'

    print('  OK: tool spec generation')


def test_agent_read_tasks():
    """Test reading tasks from file."""
    agent = Agent(model='claude-sonnet-4-6', role='test', goal='test')

    # Existing file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write('- [ ] Task 1\n- [ ] Task 2')
        f.flush()
        content = agent._read_tasks(f.name)
        assert 'Task 1' in content
        assert 'Task 2' in content
        os.unlink(f.name)

    # Non-existent file returns empty string
    assert agent._read_tasks('/nonexistent/file.md') == ''

    print('  OK: read tasks')


def test_agent_with_model_instance():
    """Test Agent accepts a Model instance directly."""
    model = Model('claude-sonnet-4-6')
    agent = Agent(model=model, role='test', goal='test')
    assert agent.model is model

    print('  OK: agent with model instance')


def test_filesystem_tool_symlink_write_blocked():
    """Test that FileSystemTool blocks writing through symlinks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        write_dir = Path(tmpdir) / 'write'
        write_dir.mkdir()
        outside = Path(tmpdir) / 'outside'
        outside.mkdir()
        link = write_dir / 'sneaky_link'
        link.symlink_to(outside)

        tool = FileSystemTool(write=[str(write_dir)])
        try:
            tool.write_file(str(link / 'evil.txt'), 'bad content')
            assert False, 'Should have raised PermissionError'
        except PermissionError:
            pass

    print('  OK: symlink write blocked')


def test_shell_tool_timeout():
    """Test ShellTool with custom timeout."""
    tool = ShellTool(allowlist=['sleep'], timeout=1)
    result = tool.run_command('sleep', ['5'])
    assert 'exit_code' in result or 'timed out' in result.lower() or 'Error' in result

    print('  OK: shell tool timeout')


def test_web_fetch_tool_blocked_schemes():
    """Test WebFetchTool blocks non-http schemes."""
    tool = WebFetchTool()
    for scheme in ['ftp://example.com', 'file:///etc/passwd', 'gopher://example.com']:
        try:
            tool.fetch_url(scheme)
            assert False, f'Should have blocked {scheme}'
        except PermissionError:
            pass

    print('  OK: web fetch blocked schemes')


# ──────────────────────────────────────────────
# Integration test (requires API key)
# ──────────────────────────────────────────────

def test_agent_run_live():
    """Integration test: run agent with a simple task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = Path(tmpdir) / 'src'
        src_dir.mkdir()
        (src_dir / 'hello.py').write_text('def greet(name):\n    print(f"Hello {name}")\n')

        tasks_file = Path(tmpdir) / 'tasks.md'
        tasks_file.write_text(
            '# Tasks\n\n'
            f'- [ ] Read the file {src_dir}/hello.py\n'
            '- [ ] Report what functions are in it\n'
        )

        agent = Agent(
            model='claude-sonnet-4-6',
            role='Code reviewer',
            goal='Review Python files',
            tools=[FileSystemTool(read=[str(src_dir)])],
            max_iterations=10,
        )

        async def run():
            events = []
            async for event in agent.run(str(tasks_file)):
                events.append(event)
                if event.type == 'response':
                    print(event.content, end='')
                elif event.type == 'tool_call':
                    print(f'\n  [tool: {event.name}]')
                elif event.type == 'done':
                    print(f'\n  Done: {event.result.iterations} iterations, {event.result.tokens} tokens')

            # Verify we got a done event
            done_events = [e for e in events if e.type == 'done']
            assert len(done_events) == 1
            result = done_events[0].result
            assert result.answer
            assert result.iterations > 0

        asyncio.run(run())

    print('  OK: live agent run')


if __name__ == '__main__':
    load_dotenv(override=True)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.disable(logging.WARNING)  # Suppress expected warnings in tests

    print('Unit tests:')
    test_dataclasses()
    test_filesystem_tool_read_write()
    test_filesystem_tool_symlink_write_blocked()
    test_shell_tool()
    test_shell_tool_timeout()
    test_web_fetch_tool_ssrf()
    test_web_fetch_tool_blocked_schemes()
    test_skills_loader()
    test_tool_specs()
    test_agent_init()
    test_agent_system_prompt()
    test_agent_system_prompt_with_skills()
    test_agent_system_prompt_with_instructions()
    test_agent_final_answer_tool()
    test_agent_execute_unknown_tool()
    test_agent_execute_tool_error()
    test_agent_audit_trail()
    test_agent_tool_decorator_with_context()
    test_agent_build_tool_specs()
    test_agent_read_tasks()
    test_agent_with_model_instance()

    print('\nIntegration tests:')
    test_agent_run_live()

    print('\nAll tests passed!')
