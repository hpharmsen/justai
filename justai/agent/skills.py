"""Skill loader: reads and concatenates .md files from a directory."""
from pathlib import Path


def load_skills(skills_dir: str | Path) -> str:
    """Load all .md files from skills_dir, concatenated alphabetically."""
    path = Path(skills_dir)
    if not path.exists():
        raise FileNotFoundError(f'Skills directory not found: {skills_dir}')
    if not path.is_dir():
        raise ValueError(f'Skills path is not a directory: {skills_dir}')

    return '\n\n'.join(f.read_text().strip() for f in sorted(path.glob('*.md')))
