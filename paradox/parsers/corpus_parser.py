import xml.etree.ElementTree
import logging


def parse(path=None, reverse=True):
    map_class_numeric = {'P': 0, 'NP': 1, 'SP': 2, 'XX': 3}  # Needed for keras
    e = xml.etree.ElementTree.parse(path).getroot()
    logging.debug('parsing data set: name={}, version={}'.format(e.get('name'), e.get('version')))
    corpus = e.findall('Corpus')[0]
    language = corpus.findall('Language')
    logging.debug("detected language: {}".format(corpus.find('Language').text))
    paraphrases = corpus.findall('Paraphrase')
    X, y, P = [], [], []
    for p in paraphrases:
        p_id = p.get("pID")
        P.append(p_id)
        sentence1 = p.find("Sentence1").text
        sentence2 = p.find("Sentence2").text
        clazz = p.find("Class").text
        X.append((sentence1, sentence2))
        # X.append((sentence2, sentence1))
        # y.append(clazz)
        # y.append(map_class_numeric[clazz])
        y.append(map_class_numeric[clazz])
        logging.debug('parsed pair with id: {}'.format(p_id))
    logging.debug('parsed {} pairs of sentences'.format(len(paraphrases)))
    return X, y, P
