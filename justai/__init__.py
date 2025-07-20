from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

from justai.model.model import Model
from justai.tools.prompts import get_prompt, set_prompt_file, add_prompt_file


def _get_version():
    try:
        __version__ = version(__name__)
    except PackageNotFoundError:
        with open(Path(__file__).parent / "pyproject.toml", "r") as f:
            for line in f:
                if line.startswith("version ="):
                    return line.split('"')[1]
            raise RuntimeError("Unable to find version")


__version__ = _get_version()

# Use like this
# from importlib.metadata import version
# print(version("justai"))

if __name__ == '__main__':
    # Onderstaande om de voorkomen dat import optimizer ze leeg gooit
    a = Model
    g = get_prompt
    s = set_prompt_file
    apf = add_prompt_file
