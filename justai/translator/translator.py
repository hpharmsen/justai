import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from justai.translator.xliff import get_xliff_version, translate_xliff_1_2, translate_xliff_2_0, is_translatable
from justai.agent.agent import Agent
from justai.tools.prompts import get_prompt, set_prompt_file


class Translator(Agent):

    def __init__(self, model=None):
        if not model:
            model = os.environ.get('OPENAI_MODEL', 'gpt-4-turbo-preview')
        super().__init__(model, temperature=0, max_tokens=4096)
        set_prompt_file(Path(__file__).parent / 'prompts.toml')
        self.system_message = get_prompt('SYSTEM')
        self.input = ''
        self.input_format = ''

    def load(self, input_file: str | Path):
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            self.read(f.read())

    def read(self, input_string: str):
        # Input bestaat uit <transunit> elementen. Die hebben een datatype property.
        # Binnen elke <transunit> zit een <source> element en komt (na vertaling) een <target> element.
        # ALs datatype == "plaintext" dan zit de te vertalen tekst direct in de <source>
        # Als datatype == "x-DocumentState" dan zit er in de <source> een <g> element met daarin de te vertalen tekst.

        # In 2.0:
        # Input bestaat uit <unit> elementen. Die hebben een Id.
        # Binnen elke <unit> zit een <segment> en daarin een <source>
        # In de source zit ofwel direct tekst, ofwel een <pc> element
        # met daarin nog een <pc> element met daarin de te vertalen tekst
        self.input = input_string
        if input_string.strip().startswith('<?xml ') and 'xliff:document:' in input_string[:500]:
            self.input_format = get_xliff_version(input_string)
        else:
            self.input_format = 'plaintext'

    def translate(self, language: str) -> str:
        if self.input_format == 'xliff 1.2':
            return translate_xliff_1_2(self.input, self.translate_multiple, language)
        elif self.input_format == 'xliff 2.0':
            return translate_xliff_2_0(self.input, self.translate_multiple, language)
        else:
            return self.translate_single(language)

    def translate_single(self, language: str) -> str:
        """ Used to translate a single string """

        # @cached
        def run_prompt(prompt: str):
            return self.chat(prompt, return_json=False)

        prompt = get_prompt('TRANSLATE_SINGLE', language=language, translate_str=self.input)
        target_str = run_prompt(prompt)
        return target_str

    def translate_multiple(self, texts: list, language: str):
        """ Used to translate a list of strings, returns a similar list, but translated """
        # @cached
        def run_prompt(prompt: str):
            return self.chat(prompt, return_json=False)

        def split_list_in_sublists(source_list, max_chunk_len):
            chunks = []
            for text in source_list:
                if not chunks or chunks[-1] and len(chunks[-1]) + len(text) > max_chunk_len:
                    chunks.append([text])
                else:
                    chunks[-1].append(text)
            return chunks

        source_list = list(set([text for text in texts if is_translatable(text)]))  # Filter out doubles

        multiprocessing = False
        if multiprocessing:
            translation_dict = {}
            sub_lists = split_list_in_sublists(source_list, 100)
            futures = []
            with ThreadPoolExecutor(max_workers=len(sub_lists)) as executor:
                for sub_list in sub_lists:
                    source_str = '\n'.join([f'{index + 1} [[{text}]]' for index, text in enumerate(sub_list)])
                    prompt = get_prompt('TRANSLATE', language=language, translate_str=source_str,
                                        count=len(source_list))
                    futures += [executor.submit(run_prompt, prompt)]
                for index, future in enumerate(futures):
                    target_list = [t.split(']]')[0] for t in future.result().split('[[')[1:]]
                    translation_dict.update(dict(zip(sub_lists[index], target_list)))
        else:
            source_str = '\n'.join([f'{index + 1} [[{text}]]' for index, text in enumerate(source_list)])
            prompt = get_prompt('TRANSLATE_MULTIPLE', language=language, translate_str=source_str,
                                count=len(source_list))
            target_str = run_prompt(prompt)
            target_list = [t.split(']]')[0] for t in target_str.split('[[')[1:]]
            translation_dict = dict(zip(source_list, target_list))
        translations = [translation_dict.get(text, text) for text in texts]

        count = 1
        for key, val in translation_dict.items():
            print(f'{count}. {key} -> {val}')
            count += 1
        return translations


if __name__ == "__main__":

    # Make sure te justai package is in the PYTHONPATH in order for Python to recognize it as a package
    # import sys
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    load_dotenv()

    def run_test(input_file: [Path | str], language: str):
        if isinstance(input_file, str):
            input_file = Path(input_file)
        tr = Translator()
        try:
            tr.load(input_file)
        except ValueError as e:
            print(e.args[0])
            return
        translated = tr.translate(language)
        outfile = f'{input_file.stem} {language}.xlf'
        with open(outfile, 'w') as f:
            f.write(translated)
        print(outfile)

    # start = time.time()
    # # run_test('AI_2.1.xlf', 'Oekraïens')
    # run_test('/Users/hp/ai/opdrachten/Autoniveau/autoniveau/short 1.2.xlf', 'Pools')
    # duration = time.time() - start
    # print(f'Duration: {duration:.2f} seconds')

    start = time.time()
    run_test('/Users/hp/ai/opdrachten/Autoniveau/autoniveau/test.txt', 'Spaans')
    duration = time.time() - start
    print(f'Duration: {duration:.2f} seconds')

    # run_test('Proefbestand 2.0.xlf', 'Engels')
