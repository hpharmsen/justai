import os
import hashlib
import pickle
import re
from pathlib import Path
from lxml import etree

from justai.agent.agent import Agent
from justai.tools.prompts import get_prompt, set_prompt_file
from justai.translator.languages import LANGUAGES


class Translator(Agent):

    def __init__(self, model=None):
        if not model:
            model = os.environ.get('OPENAI_MODEL', 'gpt-4-turbo-preview')
        super().__init__(model, temperature=0, max_tokens=4096)
        set_prompt_file(Path(__file__).parent / 'prompts.toml')
        self.system_message = get_prompt('SYSTEM')
        self.xml = ''
        self.version = ''

    def load(self, input_file: str | Path):
        with open(input_file, 'r') as f:
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
        self.xml = input_string
        self.messages = []
        try:
            self.version = self.xml.split('xliff:document:')[1].split('"')[0].split("'")[0]
        except IndexError:
            raise ValueError('No XLIFF version found in input')
        if self.version not in ['1.2', '2.0']:
            raise ValueError(f'Unsupported XLIFF version: {self.version}')

    def translate(self, language: str, string_cached: bool = False) -> str:
        if self.version == '1.2':
            return self.translate1_2(language, string_cached=string_cached)
        elif self.version == '2.0':
            return self.translate2_0(language, string_cached=string_cached)

    def translate1_2(self, language, string_cached: bool = False):
        # XML-data laden met lxml
        parser = etree.XMLParser(ns_clean=True)
        root = etree.fromstring(self.xml.encode('utf-8'), parser=parser)
        namespaces = {'ns': 'urn:oasis:names:tc:xliff:document:1.2'}

        # Verzamel alle te vertalen teksten en hun paden
        texts_to_translate = []

        # Start het verzamelproces vanuit <source> elementen en vertaal de teksten
        for trans_unit in root.xpath('.//ns:trans-unit', namespaces=namespaces):
            source = trans_unit.xpath('.//ns:source', namespaces=namespaces)[0]
            texts_to_translate.extend(collect_texts_from_element(source))

        # Vertaal met AI
        translated_texts = self.do_translate(texts_to_translate, language, string_cached=string_cached)

        # Plaats vertaalde teksten terug in nieuwe <target> elementen met behoud van structuur
        counter = [0]
        for trans_unit in root.xpath('.//ns:trans-unit', namespaces=namespaces):
            source = trans_unit.xpath('.//ns:source', namespaces=namespaces)[0]
            target = etree.Element('{urn:oasis:names:tc:xliff:document:1.2}target')
            copy_structure_with_texts(source, target, translated_texts, counter)
            trans_unit.append(target)

        # De bijgewerkte XLIFF-structuur omzetten naar een string en afdrukken
        updated_xml = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode('utf-8')
        return updated_xml

    def translate2_0(self, language, string_cached: bool = False):
        # XML-data laden met lxml
        parser = etree.XMLParser(ns_clean=True)
        root = etree.fromstring(self.xml.encode('utf-8'), parser=parser)
        namespaces = {'ns': 'urn:oasis:names:tc:xliff:document:2.0'}

        # Speciaal voor xliff 2.0: voeg de target language toe aan het root element
        language_code = LANGUAGES.get(language)
        root.attrib['trgLang'] = language_code

        # Verzamel alle te vertalen teksten en hun paden
        texts_to_translate = []

        # Start het verzamelproces vanuit <source> elementen en vertaal de teksten
        for source in root.xpath('.//ns:source', namespaces=namespaces):
            texts_to_translate.extend(collect_texts_from_element(source))

        # Vertaal met AI
        translated_texts = self.do_translate(texts_to_translate, language, string_cached=string_cached)

        # Plaats vertaalde teksten terug in nieuwe <target> elementen met behoud van structuur
        counter = [0]
        for segment in root.xpath('.//ns:segment', namespaces=namespaces):
            source = segment.xpath('.//ns:source', namespaces=namespaces)[0]
            target = etree.SubElement(segment, '{urn:oasis:names:tc:xliff:document:2.0}target')
            copy_structure_with_texts(source, target, translated_texts, counter)

        updated_xml = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode('utf-8')
        return updated_xml

    def do_translate(self, texts, language: str, string_cached=False):
        source_list = list(set([text for text in texts if is_translatable(text)]))  # Filter out doubles

        cache = StringCache(language) if string_cached else {}
        source_list = [text for text in source_list if text not in cache]

        if source_list:
            source_str = '\n'.join([f'{index + 1} [[{text}]]' for index, text in enumerate(source_list)])
            prompt = get_prompt('TRANSLATE_MULTIPLE', language=language, translate_str=source_str,
                                count=len(source_list))
            target_str = self.chat(prompt, return_json=False)
            target_list = [t.split(']]')[0] for t in target_str.split('[[')[1:]]
            translation_dict = dict(zip(source_list, target_list))

            count = 1
            for key, val in translation_dict.items():
                if key.strip() and (key[0] == ' ' or key[-1] == ' '):
                    # Code om zeker te maken dat de vertaling dezelfde whitespace aan het begin en eind heefdt heeft als de bron
                    start_spaces = (len(key) - len(key.lstrip(' '))) * ' '
                    end_spaces = (len(key) - len(key.rstrip(' '))) * ' '
                    translation_dict[key] = start_spaces + val.strip() + end_spaces
                    val = translation_dict[key]
                print(f'{count}. [{key}] -> [{val}]')
                count += 1
                ratio = len(key) / len(val)
                if ratio >= 1.5 or ratio <= 0.7:
                    print(f'Vertaling van {key} naar {val} is onverwacht lang of kort')

            cache.update(translation_dict)
            if string_cached:
                cache.save()

        translations = [cache.get(text, text) for text in texts]

        return translations

    def translate_stringlist(self, source_list, language: str, string_cached=False):
        def run_prompt(prompt: str):
            return self.chat(prompt, return_json=False)

        cache = StringCache(language) if string_cached else {}
        non_cached_list = [text for text in source_list if text not in cache and is_translatable(text)]

        if non_cached_list:
            source_str = ''
            variables = []
            for index, text in enumerate(non_cached_list):
                text_with_no_vars, vars = replace_variables_with_hash(text)
                source_str += f'{index + 1} [[{text_with_no_vars}]]\n'
                variables.extend(vars)
            prompt = get_prompt('TRANSLATE_MULTIPLE', language=language, translate_str=source_str,
                                count=len(non_cached_list))
            target_str_no_variables = run_prompt(prompt)
            print(self.input_token_count, self.output_token_count, 'tokens')
            target_str = replace_hash_with_variables(target_str_no_variables, variables)
            target_list = [t.split(']]')[0] for t in target_str.split('[[')[1:]]
            translation_dict = dict(zip(non_cached_list, target_list))
            cache.update(translation_dict)
            if string_cached:
                cache.save()
        translations = [cache.get(text, text) for text in source_list]

        return translations


def replace_variables_with_hash(text):
    # Vindt alle variabelen in de tekst
    variables = re.findall(r'%[^%]+%', text)
    # Vervang alle variabelen in de tekst met ###
    # Het model heeft moeite met newlines. Daarom vervangen we ze door @@ en na vertaling weer terug.
    modified_text = re.sub(r'%[^%]+%', '###', text).replace('\n', '@@')
    return modified_text, variables


def replace_hash_with_variables(text, variables):
    for variable in variables:
        text = text.replace('###', variable, 1)
    # en zet de newlines terug
    text = text.replace('@@', '\n')
    return text


def collect_texts_from_element(element):
    texts = []
    # if element.text and element.text.strip():
    #    texts.append(element.text.strip())
    if element.text:
        texts.append(element.text)
    for child in element:
        texts.extend(collect_texts_from_element(child))
    return texts


def copy_structure_with_texts(source, target, translated_texts, counter=[0]):
    """ Kopieer de structuur van <source> naar <target> en behoud de teksten """
    if source.text:  # and source.text.strip():
        try:
            target.text = translated_texts[counter[0]]
            counter[0] += 1
        except IndexError:
            print('IndexError in copy_structure_with_texts')
    for child in source:
        child_copy = etree.SubElement(target, child.tag, attrib=child.attrib)
        copy_structure_with_texts(child, child_copy, translated_texts, counter)


def is_translatable(text) -> bool:
    """ Returns True if the unit should be translated """
    return text and re.search('[a-zA-Z]{2}', text) and text[0] not in ('%', '<')


def split_list_in_sublists(source_list, max_chunk_len):
    chunks = []
    for text in source_list:
        if not chunks or chunks[-1] and len(chunks[-1]) + len(text) > max_chunk_len:
            chunks.append([text])
        else:
            chunks[-1].append(text)
    return chunks


class StringCache:
    def __init__(self, language: str):
        self.language = language
        self.cache = {}
        self.file = Path(__file__).parent / (self.language + '.pickle')
        try:
            with open(self.file, 'rb') as f:
                self.cache = pickle.load(f)
        except (FileNotFoundError, EOFError):
            self.cache = {}

    def get(self, source, default=None):
        key = self.get_key(source)
        return self.cache.get(key, default)

    def set(self, source, translation):
        key = self.get_key(source)
        self.cache[key] = translation

    def __contains__(self, source):
        key = self.get_key(source)
        return key in self.cache

    def update(self, translation_dict):
        for source, translation in translation_dict.items():
            self.set(source, translation)

    def save(self):
        with open(self.file, 'wb') as f:
            pickle.dump(self.cache, f)

    def clear(self):
        self.cache = {}
        self.save()

    @classmethod
    def get_key(cls, source):
        return hashlib.md5(source.encode('utf-8')).hexdigest()


def parse_xliff_with_unit_clusters(xliff_content, max_chunk_size):
    # Functie alleen gebruikt voor tests. Deze is gelijk aan de parseXLIFFWithUnitClusters uit javascript
    # en is bedoeld om de real life situatie met het splitsen van de XLIFF in clusters te simuleren.

    # Bepaal de versie van xliffContent
    version_match = re.search(r'<xliff[^>]*\s+version="([0-9.]+)"', xliff_content)
    version = version_match.group(1) if version_match else None

    # Splits xliff in header, clusters van maxChunkSize, and footer
    header_re = r'^(.*?)<unit ' if version == '2.0' else r'^(.*?)<trans-unit '
    header_match = re.match(header_re, xliff_content, re.DOTALL)
    header = header_match.group(1) if header_match else None

    # Extract units and cluster them
    units = []
    cluster = ''
    unit_re = r'<unit .*?</unit>' if version == '2.0' else r'<trans-unit .*?</trans-unit>'
    matches = re.findall(unit_re, xliff_content, re.DOTALL)

    for match in matches:
        unit = match
        if len(unit) > max_chunk_size:
            # If current unit is larger than maxChunkSize, push current cluster (if not empty) and then this unit
            if cluster:
                units.append(cluster)
                cluster = ''
            units.append(unit)
        elif len(cluster) + len(unit) > max_chunk_size:
            # If adding this unit exceeds the limit, push current cluster and start a new one
            units.append(cluster)
            cluster = unit
        else:
            # Add this unit to current cluster
            cluster += unit

    # Don't forget to add the last cluster if it exists
    if cluster:
        units.append(cluster)

    # Extract footer
    footer_re = r'</unit>' if version == '2.0' else r'</trans-unit>'
    footer = xliff_content.split(footer_re)[-1]

    return {"header": header, "units": units, "footer": footer, "version": version}
