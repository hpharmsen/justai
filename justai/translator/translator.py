import os
import hashlib
import pickle
import re
from pathlib import Path
from lxml import etree
from enum import Enum

from justai.agent.agent import Agent
from justai.tools.log import Log
from justai.tools.prompts import get_prompt, set_prompt_file
from justai.translator.languages import LANGUAGES


# Translation options
class Opt(Enum):
    REPLACE_VARIABLES = 'replace_variables'  # Vervang %variabelen% voor vertaling en zet ze daarna terug
    LOG = 'log'
    STRING_CACHED = 'string_cached'
    CONCATENATED = 'concatenated'  # Of strings zijn samengevoegd met || om ze samen te vertalen
    PROMPT = 'prompt'  # Om een eigen prompt mee te geven.


class Translator(Agent):

    def __init__(self, model=None, **kwargs):
        if not model:
            model = os.environ.get('MODEL', 'claude-3-opus-20240229')
        super().__init__(model, temperature=0, max_tokens=4096, **kwargs)
        set_prompt_file(Path(__file__).parent / 'prompts.toml')
        self.system_message = get_prompt('SYSTEM')
        self.xml = ''
        self.version = ''
        self.logger = Log()

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
        log = self.logger
        parser = etree.XMLParser(ns_clean=True)
        root = etree.fromstring(self.xml.encode('utf-8'), parser=parser)
        namespaces = {'ns': f'urn:oasis:names:tc:xliff:document:{self.version}'}

        segment_name = 'trans-unit' if self.version == '1.2' else 'segment'

        if self.version >= '2.0':
            root.attrib['trgLang'] = LANGUAGES.get(language)

        # Verzamel teksten als lijst van samengevoegde strings per <source>
        all_texts = []
        translatable = []
        for source in root.xpath('.//ns:source', namespaces=namespaces):
            texts_from_source_element = collect_texts_from_element(source)
            all_texts.append(texts_from_source_element)
            # Verzamel alleen de teksten die ook echt vertaald moeten worden
            translatable_from_source_element = [text for text in texts_from_source_element if is_translatable(text)]
            if translatable_from_source_element:
                translatable.append("||".join(translatable_from_source_element))
        log.info('translate - all_texts', all_texts)
        log.info('translate - translatable_texts', translatable)

        # Vertaal de lijst van met || samengevoegde strings
        options = {Opt.STRING_CACHED: string_cached, Opt.LOG: True, Opt.CONCATENATED: True, Opt.REPLACE_VARIABLES: True}
        translated = self.do_translate(translatable, language, options)

        # Zet nu de vertaalde delen terug in all_texts
        index_in_translated = 0
        for texts_from_source_element in all_texts:
            translatable_from_source_element = [text for text in texts_from_source_element if is_translatable(text)]
            if translatable_from_source_element:
                translated_version = translated[index_in_translated].split("||")

                # Check
                sc = len(translatable_from_source_element)
                tc = len(translated_version)
                if sc != tc:
                    log.error('translate - mismatch',
                              f'Source texts ({sc}) does not match number of translated texts ({tc})')
                    log.info('translate - translatable_from_source_element:', translatable_from_source_element)
                    log.info('translate - translated_version:', translated_version)
                assert sc == tc, 'Mismatch see log'

                for index, text in enumerate(texts_from_source_element):
                    if is_translatable(text):
                        texts_from_source_element[index] = translated_version.pop(0)
                index_in_translated += 1
        flattened_all_texts = [item for sublist in all_texts for item in sublist]  # And flatten

        # Plaats vertaalde teksten in nieuwe <target> elementen met behoud van structuur
        counter = [0]
        for segment in root.xpath(f'.//ns:{segment_name}', namespaces=namespaces):
            source = segment.xpath('.//ns:source', namespaces=namespaces)[0]
            target = etree.SubElement(segment, f'{{urn:oasis:names:tc:xliff:document:{self.version}}}target')
            copy_structure_with_texts(source, target, flattened_all_texts, counter)

        updated_xml = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode('utf-8')
        return updated_xml

    def do_translate(self, texts, language: str, options: dict = {}):
        # Options can be:
        # REPLACE_VARIABLES: True | False
        # LOG: True | False
        # STRING_CACHED: True | False
        # CONCATENATED: True | False
        # PROMPT = 'prompt'  # Om een eigen prompt mee te geven.

        # assert all(is_translatable(text) for text in texts), "Not all translatable"  # !! Tijdelijk
        cache = StringCache(language) if options.get(Opt.STRING_CACHED) else {}
        source_list = [text for text in texts if text not in cache]

        if source_list:
            variables = []
            source_str_no_vars = ''
            for index, text in enumerate(source_list):
                if options.get(Opt.REPLACE_VARIABLES):
                    text_no_vars, v = replace_variables_with_hash(text)
                    variables.extend(v)
                else:
                    text_no_vars = text
                source_str_no_vars += f'{index + 1} [[{text_no_vars}]]\n'
            given_prompt = options.get(Opt.PROMPT)
            if given_prompt:
                prompt = given_prompt.format(language=language, translate_str=source_str_no_vars,
                                             count=len(source_list))
            else:
                prompt = get_prompt('TRANSLATE_MULTIPLE', language=language, translate_str=source_str_no_vars,
                                    count=len(source_list))
            target_str_no_vars = self.chat(prompt, return_json=False, cached=False)
            if options.get(Opt.REPLACE_VARIABLES):
                target_str = replace_hash_with_variables(target_str_no_vars, variables)
            else:
                target_str = target_str_no_vars
            if options.get(Opt.LOG):
                log = self.logger
                log.response('target_str_no_vars', target_str_no_vars)
                log.prompt('prompt', prompt)
                log.info('do_translate - target_str', target_str)

            target_list = [t.split(']]')[0] for t in target_str.split('[[')[1:]]

            count = 1
            for source, translation in zip(source_list, target_list):

                if options.get(Opt.CONCATENATED):
                    source_parts = source.split('||')
                    translation_parts = translation.split('||')
                    # Check op het aantal onderdelen tussen ||. Dit moet gelijk zijn in de bron en de vertaling
                    sc = len(source_parts)
                    tc = len(translation_parts)
                    if tc != sc:
                        log.warning('do_translate',
                                    f'Number of translated texts ({tc}) does not match number of source texts ({sc}). Correcting.')
                        # Model krijgt het niet voor elkaar om een zin met evenveel delen terug te geven
                        # We vertalen de stukjes los van elkaar.
                        translation_parts = self.do_translate(source_parts, language, options)
                        # if tc < sc:
                        #     translation_parts += [' '] * (sc - tc)
                        # else:
                        #     # Dit is een hack! Het model geeft kennelijk meer ||'s terug in dan in het origineel.
                        #     # Dat breekt de code verderop. Daarom voegen we de laatste onderdelen samen.
                        #     # Maar dat levert wel een onjuiste vertaling op.
                        #     translation_parts = translation_parts[:sc - 1] + [' '.join(translation_parts[sc - 1:])]
                        sc = len(source_parts)
                        tc = len(translation_parts)
                        if tc != sc:
                            log.error('do_translate',
                                      f'Number of translated texts ({tc}) still does not match number of source texts ({sc}). Correcting.')
                        assert (tc == sc), 'Mismatch in number of parts, see log for info'
                else:
                    source_parts = [source]
                    translation_parts = [translation]

                # Zorg dat de vertaling dezelfde whitespace aan het begin en eind heeft als de bron
                for i, (source_part, translation_part) in enumerate(zip(source_parts, translation_parts)):
                    if source_part.strip() and (source_part[0] == ' ' or source_part[-1] == ' '):
                        start_spaces = (len(source_part) - len(source_part.lstrip(' '))) * ' '
                        end_spaces = (len(source_part) - len(source_part.rstrip(' '))) * ' '
                        translation_parts[i] = start_spaces + translation_part.strip() + end_spaces

                target_list[count - 1] = '||'.join(translation_parts)

                count += 1

                if options.get(Opt.LOG):
                    ratio = len(source) / len(translation)
                    if ratio >= 1.5 or ratio <= 0.7:
                        log.warning('', f'Vertaling van {source} naar {translation} is onverwacht lang of kort')

            translation_dict = dict(zip(source_list, target_list))
            cache.update(translation_dict)
            if options.get(Opt.STRING_CACHED):
                cache.save()

        translations = [cache.get(text, text) for text in texts]

        if options.get(Opt.CONCATENATED):
            for s, t in zip(translations, texts):
                sparts, tparts = s.split('||'), t.split('||')
                if len(sparts) != len(tparts):
                    assert False, f'Number of parts in source ({len(sparts)}) does not match number of parts in translation ({len(tparts)})'

        return translations

    def translate_dict(self, text_dict, language: str, options):
        return {k: v for k, v in zip(text_dict.keys(),
                                     self.do_translate(list(text_dict.values()), language, options))}


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


def copy_structure_with_texts(source, target, translated_texts, counter=[0], log=None):
    """ Kopieer de structuur van <source> naar <target> en behoud de teksten """
    if source.text:  # and source.text.strip():
        try:
            target.text = translated_texts[counter[0]]
            counter[0] += 1
        except IndexError:
            if log:
                log.error('IndexError', 'IndexError in copy_structure_with_texts')
    for child in source:
        child_copy = etree.SubElement(target, child.tag, attrib=child.attrib)
        copy_structure_with_texts(child, child_copy, translated_texts, counter, log)


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

    def __len__(self):
        return len(self.cache)

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
