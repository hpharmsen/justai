"""Tests for the Agent class, built-in tools, and skills loader.

Usage:
    python tests/test_agent.py
"""
import asyncio
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from justai import Agent, AgentEvent, AgentResult, FileSystemTool, ShellTool, WebFetchTool
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
    ctx = __import__('justai.agent.agent', fromlist=['AgentContext']).AgentContext(deps={'user': 'test'})
    prompt = agent._build_system_prompt(ctx)
    assert 'developer' in prompt
    assert 'fix bugs' in prompt
    assert 'final_answer' in prompt

    print('  OK: system prompt composition')


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

    print('Unit tests:')
    test_dataclasses()
    test_filesystem_tool_read_write()
    test_shell_tool()
    test_web_fetch_tool_ssrf()
    test_skills_loader()
    test_tool_specs()
    test_agent_init()
    test_agent_system_prompt()

    print('\nIntegration tests:')
    test_agent_run_live()

    print('\nAll tests passed!')
