from pathlib import Path

from dotenv import load_dotenv

from justai.agent.agent import Agent
from justai.tools.prompts import add_prompt_file, get_prompt


class Translator(Agent):

    def __init__(self, model):
        super().__init__(model, temperature=0.3)
        add_prompt_file(Path(__file__).parent / 'prompts.toml')
        self.system_message = get_prompt('SYSTEM')

        self.header = ''
        self.units = []
        self.footer = ''


    def load(self, input_file: str|Path):
        # Input bestaat uit <transunit> elementen. Die hebben een datatype property.
        # Binnen elke <transunit> zit een <source> element en komt (na vertaling) een <target> element.
        # ALs datatype == "plaintext" dan zit de te vertalen tekst direct in de <source>
        # Als datatype == "x-DocumentState" dan zit er in de <source> een <g> element met daarin de te vertalen tekst.
        with open(input_file, 'r') as f:
            return self.read(f.read())

    def read(self, xml: str):
        if not 'urn:oasis:names:tc:xliff:document:1.2' in xml[:300]:
            raise ValueError('Not an XLIFF 1.2 file')
        transunit_blocks = xml.split('<trans-unit')
        self.header = transunit_blocks[0]
        transunit_blocks = transunit_blocks[1:]
        transunit_blocks[-1], self.footer = transunit_blocks[-1].split('</trans-unit>')
        transunit_blocks[-1] += '</trans-unit>'
        self.units = []
        for i in range(len(transunit_blocks)):
            tub = transunit_blocks[i]
            id = tub.split('id="')[1].split('"')[0]
            xml_space = tub.split('xml:space="')[1].split('"')[0] if 'xml:space="' in tub else None
            datatype = tub.split('datatype="')[1].split('"')[0]
            source = tub.split('<source>')[1].split('</source>')[0]
            match datatype:
                case "x-DocumentState":
                    texts = []
                    for block in source.split('<g ')[1:]:
                        t = block.split('">')[1].split('</g>')[0]
                        if _translatable({'text': t}):
                            texts += [t]
                    text = '||'.join(texts)
                    #text = '||'.join([block.split('">')[1].split('</g>')[0] for block in source.split('<g ')[1:]])
                case "plaintext":
                    text = source if _translatable({'text': source}) else ""
                case _:
                    raise ValueError(f'Unknown datatype {datatype}')

            self.units += [{'id': id,
                       'xml_space': xml_space,
                       'datatype': datatype,
                       'source': source,
                       'text': text}] # text is "" als er niks te vertalen is

    def write(self):
        xml = self.header
        for unit in self.units:
            xml_space = f'xml:space="{unit["xml_space"]}" ' if unit.get('xml_space') else ''
            xml += f'''<trans-unit id="{unit['id']}" {xml_space}datatype="{unit['datatype']}">\n'''
            xml += f'''				<source>{unit['source']}</source>\n'''
            xml += '                <target>'
            if unit.get('target'):
                xml += unit['target']
            else:
                xml += unit['source']
            xml += '</target>\n'
            xml += "</trans-unit>\n"
        xml += self.footer
        return xml

    def save_xlf(self, output_file: str | Path):
        with open(output_file, 'w') as f:
            f.write(self.write())

    def translate(self, language: str):
        translatable_units = [u for u in self.units if u['text']]
        translate_list = [f"{index+1} [[{unit['text']}]]\n" for index, unit in enumerate(translatable_units)]
        # TODO: translate_str stap voor stap opbouwen tot ie over de 3000 tekens gaat.
        #  Dan prompt, toevoegen aan translations en repeat
        translate_str = "".join(translate_list)
        prompt = get_prompt('TRANSLATE', language=language, translate_str=translate_str, count=len(translate_list))
        prompt_result = self.chat(prompt, return_json=False)
        translations = [t.split(']]')[0] for t in prompt_result.split('[[')[1:]]
        index = 0
        for unit in self.units:
            if unit['text']:
                translation = translations[index]
                unit['translation'] = translation
                index += 1
                text = unit['text']
                print(text,'->',translation, '\n')
                unit['target'] = unit['source']
                for s, t in zip(text.split('||'), translation.split('||'), strict=True):
                    unit['target'] = unit['target'].replace('>' + s + '<', '>' + t + '<')


def _translatable(unit: dict) -> bool:
    """ Returns True if the unit should be translated """
    return unit['text'] and len(unit['text']) > 1 and unit['text'][0] != '%'


if __name__ == "__main__":
    load_dotenv()
    tr = Translator('gpt-4-turbo-preview')
    tr.load('/Users/hp/ai/opdrachten/Autoniveau/autoniveau/AI.xlf')
    tr.translate('Engels')
    tr.save_xlf('/Users/hp/Downloads/AI_en.xlf')
