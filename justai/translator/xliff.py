from lxml import etree

from justai.translator.languages import LANGUAGES


def get_xliff_version(input_string):
    try:
        xlif_version = input_string.split('xliff:document:')[1].split('"')[0].split("'")[0]
    except IndexError:
        raise ValueError(f'No XLIFF version found in input')
    if xlif_version not in ['1.2', '2.0']:
        raise ValueError(f'Unsupported XLIFF version: {xlif_version}')
    return 'xliff ' + xlif_version


def translate_xliff_1_2(source_xml: str, translate_function, language: str):
    texts_to_translate = xliff_12_to_string_list(source_xml)
    translated_texts, rest = translate_function(texts_to_translate, language)
    return xliff_12_add_translations(source_xml, translated_texts, language)


def xliff_12_to_string_list(source_xml:str) -> list[str]:
    parser = etree.XMLParser(ns_clean=True)
    root = etree.fromstring(source_xml.encode('utf-8'), parser=parser)
    namespaces = {'ns': 'urn:oasis:names:tc:xliff:document:1.2'}

    # Verzamel alle te vertalen teksten en hun paden
    texts_to_translate = []

    # Start het verzamelproces vanuit <source> elementen en vertaal de teksten
    for trans_unit in root.xpath('.//ns:trans-unit', namespaces=namespaces):
        source = trans_unit.xpath('.//ns:source', namespaces=namespaces)[0]
        texts_to_translate.extend(collect_texts_from_element(source))
    return texts_to_translate


def xliff_12_add_translations(source_xml:str, translated_texts:list[str], language:str) -> str:
    parser = etree.XMLParser(ns_clean=True)
    root = etree.fromstring(source_xml.encode('utf-8'), parser=parser)
    namespaces = {'ns': 'urn:oasis:names:tc:xliff:document:1.2'}

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


def translate_xliff_2_0(source_xml:str, translate_function, language:str) -> str:
    texts_to_translate = xliff_20_to_string_list(source_xml)
    translated_texts, rest = translate_function(texts_to_translate, language)
    return xliff_20_add_translations(source_xml, translated_texts, language)


def xliff_20_to_string_list(source_xml:str) -> list[str]:
    parser = etree.XMLParser(ns_clean=True)
    root = etree.fromstring(source_xml.encode('utf-8'), parser=parser)
    namespaces = {'ns': 'urn:oasis:names:tc:xliff:document:2.0'}

    # Verzamel alle te vertalen teksten en hun paden
    texts_to_translate = []

    # Start het verzamelproces vanuit <source> elementen en vertaal de teksten
    for source in root.xpath('.//ns:source', namespaces=namespaces):
        texts_to_translate.extend(collect_texts_from_element(source))
    return texts_to_translate


def xliff_20_add_translations(source_xml:str, translated_texts:list[str], language:str) -> str:
    parser = etree.XMLParser(ns_clean=True)
    root = etree.fromstring(source_xml.encode('utf-8'), parser=parser)
    namespaces = {'ns': 'urn:oasis:names:tc:xliff:document:2.0'}

    # Speciaal voor xliff 2.0: voeg de target language toe aan het root element
    language_code = LANGUAGES.get(language)
    root.attrib['trgLang'] = language_code

    # Plaats vertaalde teksten terug in nieuwe <target> elementen met behoud van structuur
    counter = [0]
    for segment in root.xpath('.//ns:segment', namespaces=namespaces):
        source = segment.xpath('.//ns:source', namespaces=namespaces)[0]
        target = etree.SubElement(segment, '{urn:oasis:names:tc:xliff:document:2.0}target')
        copy_structure_with_texts(source, target, translated_texts, counter)

    # De bijgewerkte XLIFF-structuur omzetten naar een string en afdrukken
    updated_xml = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode('utf-8')
    return updated_xml


def collect_texts_from_element(element):
    texts = []
    if element.text and element.text.strip():
        texts.append(element.text)
    for child in element:
        texts.extend(collect_texts_from_element(child))
    return texts


def copy_structure_with_texts(source, target, translated_texts, counter=[0]):
    """ Kopieer de structuur van <source> naar <target> en behoud de teksten """
    if source.text and source.text.strip():
        try:
            target.text = translated_texts[counter[0]]
            counter[0] += 1
        except IndexError:
            print('IndexError')
    for child in source:
        child_copy = etree.SubElement(target, child.tag, attrib=child.attrib)
        copy_structure_with_texts(child, child_copy, translated_texts, counter)


def is_translatable(text) -> bool:
    """ Returns True if the unit should be translated """
    return text and len(text.strip()) > 1 and text[0] not in ('%', '<')
