from ..client.redis_client import RedisClient
from dragnet import content_extractor
from collections import OrderedDict
from ..client.fetcher import fetch
from bs4 import BeautifulSoup
import logging
import os.path
import time
import json
import io


def download(lang='de', path='corpus/'):
    """downloads the news in redis and write them to json files.
    For each news object a json file which has the id of news is created

    # Arguments:
        lang: language of the corpus
    """
    redis_client = RedisClient(lang=lang)
    for news in populate(redis_client):
        if not is_valid(news, field='headline'):
            continue
        filename = news['id'] + '.json'
        dest = os.path.join(path, lang, 'docs')
        if not os.path.exists(dest):
            os.makedirs(dest)
        with io.open(os.path.join(dest, filename),
                     'w', encoding='utf8') as json_file:
            data = json.dumps(news,
                              ensure_ascii=False,
                              encoding='utf8',
                              indent=4)
            logging.info('Wrote document to disk: id={}'.format(news['id']))
            json_file.write(unicode(data))


def populate(redis_client):
    """Populates the entries in the database with fields such as headline,
    body, html and url

    # Arguments
        lang: language of the database

    # Returns
        news: news objects populated with required fields
    """
    keys = redis_client.keys()
    folder = 'docs/{}/'.format(redis_client.lang)
    for key in keys:
        value = redis_client.get(key)
        f = folder + value['id'] + '.json'
        if os.path.isfile(f):
            logging.info('Skipping existing document: {}'.format(f))
            continue
        if value['wayback_url'] == 'None':
            html = fetch(value['url'])
        else:
            html = fetch(value['wayback_url'])
        time.sleep(1)
        if html:
            soup = BeautifulSoup(html, 'html.parser')
        else:
            continue
        headline_elems = soup.select(value['headline_selector'], None)
        if len(headline_elems) > 0:
            headline = headline_elems[0].text.strip()
        else:
            logging.debug('Headline can not be refound: url={}, selector={}'
                          .format(value['url'], value['headline_selector']))
            continue
        news = OrderedDict()
        news['id'] = value['id']
        news['timestamp'] = value['timestamp']
        news['lang'] = redis_client.lang
        news['url'] = value['url']
        news['wayback_url'] = value['wayback_url']
        news['headline'] = headline.strip()
        news['body'] = content_extractor.analyze(html).strip()
        yield news


def is_valid(news, field=None):
    """Checks fields in a news object for validity. If a field does not exist,
    or its value is not defined, return False.

    # Arguments
        news: a news dictionary object
        field: field to be checked for validity

    # Returns:
        valid: returns true if field is valid
    """
    try:
        news[field]
    except:
        return False
    if news[field]:
        return True
    return False
