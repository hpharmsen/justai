"""Demonstrates the Agent class API.

Shows how to create an agent with tools, skills, custom tools,
and dynamic instructions. Assumes justai.Agent is implemented.
"""
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from justai import Agent, FileSystemTool, ShellTool, WebFetchTool


async def basic_example():
    """Minimal agent with file access."""
    agent = Agent(
        model='claude-sonnet-4-6',
        role='Python developer',
        goal='Improve code quality by adding type hints and docstrings',
        tools=[
            FileSystemTool(read=['./src'], write=['./output']),
        ],
    )

    # Streaming — get events as the agent works
    async for event in agent.run('tasks.md'):
        if event.type == 'status':
            print(f'[{event.message}]')
        elif event.type == 'response':
            print(event.content, end='')
        elif event.type == 'tool_call':
            print(f'Tool: {event.name}({event.arguments}) → {event.tool_result[:80]}')
        elif event.type == 'done':
            print(f'\nDone in {event.result.iterations} iterations, {event.result.tokens} tokens')


async def full_example():
    """Agent with multiple tools, custom tools, dynamic instructions, and deps."""
    agent = Agent(
        model='claude-sonnet-4-6',
        role='Senior developer',
        goal='Fix all failing tests',
        skills_dir='./skills',
        tools=[
            FileSystemTool(read=['./src', './tests'], write=['./src']),
            ShellTool(allowlist=['python', 'pytest']),
            WebFetchTool(),
        ],
        max_iterations=20,
        verbose=False,  # Only status + done events
    )

    # Custom tool with context access
    @agent.tool
    def get_project_info(ctx) -> str:
        """Get information about the current project."""
        return f'Project: {ctx.deps["project_name"]}, Python {ctx.deps["python_version"]}'

    # Dynamic instruction — evaluated once per run()
    @agent.instructions
    def inject_context(ctx) -> str:
        return f'Working on project: {ctx.deps["project_name"]}'

    # Simple usage — just get the final result
    result = await agent.run_until_done(
        'tasks.md',
        deps={'project_name': 'myapp', 'python_version': '3.12'}
    )

    print(f'Answer: {result.answer}')
    print(f'Iterations: {result.iterations}')
    print(f'Tokens: {result.tokens}')
    print(f'Tool calls:')
    for entry in result.audit:
        status = '✓' if entry.success else '✗'
        print(f'  {status} {entry.tool_name} ({entry.duration_ms}ms)')


async def demo():
    """Run a live demo with a temp directory."""
    demo_dir = Path('/tmp/agent_demo')
    demo_dir.mkdir(exist_ok=True)
    (demo_dir / 'src').mkdir(exist_ok=True)
    (demo_dir / 'output').mkdir(exist_ok=True)

    src_dir = demo_dir / 'src'
    out_dir = demo_dir / 'output'

    (src_dir / 'hello.py').write_text(
        'def greet(name):\n    print(f"Hello {name}")\n'
    )
    (demo_dir / 'tasks.md').write_text(
        '# Tasks\n\n'
        f'- [ ] Read the file {src_dir}/hello.py\n'
        f'- [ ] Write an improved version to {out_dir}/hello.py that adds type hints and a docstring\n'
        '- [ ] Report what you did\n'
    )

    agent = Agent(
        model='claude-sonnet-4-6',
        role='Python developer',
        goal='Improve code quality',
        tools=[
            FileSystemTool(
                read=[str(src_dir)],
                write=[str(out_dir)]
            ),
        ],
        max_iterations=10,
    )

    async for event in agent.run(str(demo_dir / 'tasks.md')):
        if event.type == 'status':
            print(f'\n[{event.message}]')
        elif event.type == 'response':
            print(event.content, end='')
        elif event.type == 'tool_call':
            print(f'\n  >> Tool: {event.name}({event.arguments})')
            print(f'     Result: {event.tool_result[:120]}')
        elif event.type == 'done':
            result = event.result
            print(f'\n\n--- Done ---')
            print(f'Iterations: {result.iterations}')
            print(f'Tokens: {result.tokens}')
            print(f'Audit trail:')
            for entry in result.audit:
                status = 'OK' if entry.success else 'FAIL'
                print(f'  [{status}] {entry.tool_name} ({entry.duration_ms}ms)')

    output_file = demo_dir / 'output' / 'hello.py'
    if output_file.exists():
        print(f'\nGenerated file:\n{output_file.read_text()}')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(override=True)
    # asyncio.run(basic_example())
    # asyncio.run(full_example())
    asyncio.run(demo())
