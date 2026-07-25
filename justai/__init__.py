from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

from justai.model.model import Model
from justai.models.basemodel import (
    ConnectionException,
    AuthorizationException,
    ModelOverloadException,
    RatelimitException,
    BadRequestException,
    TimeoutException,
    GeneralException,
    RefusalException,
    EffortDownmapWarning,
)
from justai.agent import Agent, AgentEvent, AgentResult, AuditEntry, AgentContext
from justai.agent.tools import FileSystemTool, ShellTool, WebFetchTool
from justai.tools.prompts import get_prompt, set_prompt_file, add_prompt_file


def _get_version() -> str:
    try:
        return version(__name__)
    except PackageNotFoundError:
        # Running from a source checkout: read the version straight out of pyproject.toml
        with open(Path(__file__).parent / 'pyproject.toml') as f:
            for line in f:
                if line.startswith('version ='):
                    return line.split('"')[1]
        raise RuntimeError('Unable to find version')


__version__ = _get_version()

# Explicit public API. Without this, every re-export above reads as an unused import.
__all__ = [
    '__version__',
    'Model',
    'ConnectionException',
    'AuthorizationException',
    'ModelOverloadException',
    'RatelimitException',
    'BadRequestException',
    'TimeoutException',
    'GeneralException',
    'RefusalException',
    'EffortDownmapWarning',
    'Agent',
    'AgentEvent',
    'AgentResult',
    'AuditEntry',
    'AgentContext',
    'FileSystemTool',
    'ShellTool',
    'WebFetchTool',
    'get_prompt',
    'set_prompt_file',
    'add_prompt_file',
]
