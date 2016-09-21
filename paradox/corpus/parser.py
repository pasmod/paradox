import logging
import json
import os


def parse(lang='de'):
    """Parses the pairs corpus stored in path
    as a generator.

    # Arguments
        lang: currently de or en

    # Yields:
        (s1, s2, label)
    """
    if lang == 'de':
        path = 'corpus/de/pairs'
    elif lang == 'en':
        path = 'corpus/en/pairs'
    else:
        raise ValueError("Language {} is not supported.".format(lang))
    logging.info("Parsing corpus: {}".format(path))
    for root, dirs, files in os.walk(path):
        for f in files:
            filename = os.path.join(path, f)
            with open(filename) as data_file:
                pairs = json.load(data_file)["pairs"]
                logging.info("Parsed {} sentence pairs".format(
                    len(pairs)))
                for pair in pairs:
                    yield (pair["sentences"][0],
                           pair["sentences"][1],
                           pair["label"])
