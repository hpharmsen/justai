"""Skill loader: reads and concatenates .md files from a directory."""
from pathlib import Path


def load_skills(skills_dir: str | Path) -> str:
    """Load all .md files from skills_dir, concatenated alphabetically."""
    path = Path(skills_dir)
    if not path.exists():
        raise FileNotFoundError(f'Skills directory not found: {skills_dir}')
    if not path.is_dir():
        raise ValueError(f'Skills path is not a directory: {skills_dir}')

    md_files = sorted(path.glob('*.md'))
    if not md_files:
        return ''

    parts = []
    for f in md_files:
        parts.append(f.read_text().strip())
    return '\n\n'.join(parts)
