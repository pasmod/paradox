from collections import OrderedDict
from ..utils.utils import sentence_tokenizer
from random import shuffle
import logging
import io
import json
import os


def construct(src='corpus/de/docs', dest='corpus/de/pairs', lang='de'):
    """Uses the JSON documents in src and create a
    corpus of sentence pairs in dest.

    # Arguments
        src: source folder containing the JSON documents
        dest: destination folder for sentence pairs
        lang: required language
    """
    if lang == 'en':
        src = 'corpus/en/docs'
        dest = 'corpus/en/pairs'
    if not os.path.exists(dest):
        os.makedirs(dest)
    for root, dirs, files in os.walk(src):
            for f in files:
                _construct(os.path.join(src, f), dest, lang)


def _construct(filename, dest=None, lang=None):
    with open(filename) as data_file:
        news = json.load(data_file)
        tokenizer = sentence_tokenizer(lang=lang)
        news_t = _transform(news, tokenizer)
        if news_t and len(news_t["pairs"]) > 0:
            _write_json(news_t, dest)


def _write_json(dictionary, dest):
    filename = str(dictionary["id"]) + '.json'
    with io.open(os.path.join(dest, filename),
                 'w', encoding='utf8') as json_file:
        data = json.dumps(dictionary,
                          ensure_ascii=False,
                          encoding='utf8',
                          indent=4)
        logging.info('Wrote document to disk: id={}'.format(dictionary['id']))
        json_file.write(unicode(data))


def _transform(news, tokenizer):
    body = news["body"]
    sentences = tokenizer.tokenize(body)
    headline = news["headline"]
    news_t = OrderedDict({"id": news["id"],
                          "lang": news["lang"],
                          "url": news["url"] or news["wayback_url"]})
    news_t['pairs'] = _construct_pairs(sentences, headline)
    return news_t


def _construct_pairs(sentences, headline):
    min_num_sentences = 10
    pairs = []
    if sentences and len(sentences) >= min_num_sentences:
        first_sentence = sentences[0]
        pairs.append({"sentences":
                      [first_sentence, headline],
                      "label": True})
        pairs.append({"sentences":
                      [headline, first_sentence],
                      "label": True})
        pairs.append({"sentences":
                      [first_sentence, first_sentence],
                      "label": True})
        pairs.append({"sentences":
                      [headline, headline],
                      "label": True})
        shuffle(sentences)
        pairs.append({"sentences":
                      [sentences[0], sentences[1]],
                      "label": False})
        pairs.append({"sentences":
                      [sentences[1], sentences[0]],
                      "label": False})
        pairs.append({"sentences":
                      [sentences[2], sentences[3]],
                      "label": False})
        pairs.append({"sentences":
                      [sentences[3], sentences[2]],
                      "label": False})
    return pairs
