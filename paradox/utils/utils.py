import nltk


def sentence_tokenizer(lang='de'):
    """Based on the given languages, returns an NLTK
    sentence splitter (tokenizer)

    # Arguments
        lang: required languages

    # Returns
        sentence_tokenizer: NLTK sentence splitter
    """
    if lang == 'de':
        return nltk.data.load('tokenizers/punkt/german.pickle')
    elif lang == 'en':
        return nltk.data.load('tokenizers/punkt/english.pickle')
    else:
        raise ValueError("Language {} is not supported")
