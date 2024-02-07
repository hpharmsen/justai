from pathlib import Path

from dotenv import load_dotenv

from justai import Translator

if __name__ == "__main__":
    load_dotenv()
    tr = Translator('gpt-3.5-turbo')
    tr.system_message = "You are a translator who translates texts for highscool students from Dutch to other languages"
    cur_path = Path(__file__).parent.resolve()
    tr.load(cur_path / 'translation_source.xlf')
    tr.translate('Engels')
    tr.save_xlf(cur_path / 'translation_target.xlf')
