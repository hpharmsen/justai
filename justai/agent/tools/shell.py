"""ShellTool with command injection prevention."""
import subprocess


SHELL_METACHARACTERS = set(';|&&$`\\(){}[]!><\n')


class ShellTool:
    def __init__(self, allowlist: list[str] | None = None, timeout: int = 60):
        self.allowlist = set(allowlist or [])
        self.timeout = timeout

    def run_command(self, executable: str, args: list[str] | None = None) -> str:
        """Run a command with the given executable and arguments."""
        if executable not in self.allowlist:
            raise PermissionError(
                f'Executable not allowed: {executable}. Allowed: {", ".join(sorted(self.allowlist))}'
            )
        args = args or []
        # Reject arguments containing shell metacharacters
        for arg in args:
            if any(c in SHELL_METACHARACTERS for c in arg):
                raise PermissionError(f'Argument contains shell metacharacters: {arg!r}')

        try:
            result = subprocess.run(
                [executable, *args],
                capture_output=True, text=True, shell=False,
                timeout=self.timeout
            )
            output = f'exit_code: {result.returncode}'
            if result.stdout:
                output += f'\nstdout: {result.stdout}'
            if result.stderr:
                output += f'\nstderr: {result.stderr}'
            return output
        except subprocess.TimeoutExpired:
            return f'Command timed out after {self.timeout} seconds'
        except FileNotFoundError:
            return f'Executable not found: {executable}'

    def get_tools(self) -> list[tuple]:
        """Return tool specs as (name, description, parameters, callable) tuples."""
        return [(
            'run_command',
            f'Run a command. Allowed executables: {", ".join(sorted(self.allowlist))}.',
            {'executable': str, 'args': list},
            self.run_command
        )]
