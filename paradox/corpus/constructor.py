from collections import OrderedDict
from ..utils.utils import sentence_tokenizer
import logging
import io
import json
import os


def construct(src='corpus/de/docs', dest='corpus/de/pairs', lang='de'):
    """Used the JSON documents in src and create a
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
    news_t = OrderedDict({"id": news["id"],
                          "lang": news["lang"],
                          "url": news["url"] or news["wayback_url"]})
    news_t['pairs'] = _construct_pairs(sentences)
    return news_t


def _construct_pairs(sentences):
    pairs = []
    if len(sentences) <= 2:
        return pairs
    for i in range(0, 2 - 1):
        sentence_i = sentences[i].split(" ")
        sentence_ii = sentences[i+1].split(" ")
        if len(sentence_i) > 5 and len(sentence_i) < 60 and\
           len(sentence_ii) > 5 and len(sentence_ii) < 60:
            pairs.append({"sentences":
                          [sentences[i], sentences[i+1]],
                          "label": True})
            pairs.append({"sentences":
                          [sentences[i+1], sentences[i]],
                          "label": False})
        else:
            print(len(sentences[i].split(" ")))
            print(len(sentences[i+1].split(" ")))
    return pairs
