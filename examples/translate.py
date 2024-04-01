from pathlib import Path

from dotenv import load_dotenv

from justai import Translator

if __name__ == "__main__":
    load_dotenv()
    tr = Translator('gpt-3.5-turbo')
    tr.system = "You are a translator who translates texts for highscool students from Dutch to other languages"
    cur_path = Path(__file__).parent.resolve()
    tr.load(cur_path / 'translation_source.xlf')
    translated = tr.translate('Engels', string_cached=False)
    with open(cur_path / 'translation_target.xlf', 'w') as f:
        f.write(translated)
