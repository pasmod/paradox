import xml.etree.ElementTree
import logging


def parse(path=None):
    map_class_numeric = {'P': 0, 'NP': 1, 'SP': 2} # Needed for keras
    e = xml.etree.ElementTree.parse(path).getroot()
    logging.info('parsing data set: name={}, version={}'.format(e.get('name'), e.get('version')))
    corpus = e.findall('Corpus')[0]
    language = corpus.findall('Language')
    logging.info("detected language: {}".format(corpus.find('Language').text))
    paraphrases = corpus.findall('Paraphrase')
    X, y = [], []
    for p in paraphrases:
        p_id = p.get("pID")
        sentence1 = p.find("Sentence1").text
        sentence2 = p.find("Sentence2").text
        clazz = p.find("Class").text
        X.append((sentence1, sentence2))
        # y.append(clazz)
        y.append(map_class_numeric[clazz])
        logging.debug('parsed pair with id: {}'.format(p_id))
    logging.info('parsed {} pairs of sentences'.format(len(paraphrases)))
    return X, y
