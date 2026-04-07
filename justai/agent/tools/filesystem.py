"""FileSystemTool with path traversal prevention."""
from pathlib import Path


class FileSystemTool:
    def __init__(self, read: list[str] | None = None, write: list[str] | None = None):
        self.read_paths = [Path(p).resolve() for p in (read or [])]
        self.write_paths = [Path(p).resolve() for p in (write or [])]

    def _check_read(self, path: Path) -> Path:
        """Resolve path and verify it's within allowed read directories."""
        resolved = path.resolve()
        if not any(resolved.is_relative_to(allowed) for allowed in self.read_paths):
            raise PermissionError(f'Read access denied: {path} is not within allowed directories')
        return resolved

    def _check_write(self, path: Path) -> Path:
        """Resolve path and verify it's within allowed write directories, reject symlinks."""
        resolved = path.resolve()
        if not any(resolved.is_relative_to(allowed) for allowed in self.write_paths):
            raise PermissionError(f'Write access denied: {path} is not within allowed directories')
        if path.is_symlink():
            raise PermissionError(f'Write access denied: {path} is a symlink')
        return resolved

    def read_file(self, path: str) -> str:
        """Read a file and return its contents."""
        resolved = self._check_read(Path(path))
        return resolved.read_text()

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file."""
        resolved = self._check_write(Path(path))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f'Written {len(content)} bytes to {path}'

    def list_directory(self, path: str) -> str:
        """List files and directories at the given path."""
        resolved = self._check_read(Path(path))
        if not resolved.is_dir():
            raise ValueError(f'Not a directory: {path}')
        entries = sorted(resolved.iterdir())
        return '\n'.join(
            f'{"[dir]  " if e.is_dir() else "[file] "}{e.name}' for e in entries
        )

    def get_tools(self) -> list[tuple]:
        """Return tool specs as (name, description, parameters, callable) tuples."""
        tools = []
        if self.read_paths:
            tools.append((
                'read_file', 'Read a file and return its contents.',
                {'path': str}, self.read_file
            ))
            tools.append((
                'list_directory', 'List files and directories at the given path.',
                {'path': str}, self.list_directory
            ))
        if self.write_paths:
            tools.append((
                'write_file', 'Write content to a file.',
                {'path': str, 'content': str}, self.write_file
            ))
        return tools
