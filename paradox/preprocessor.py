from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

stopwords = set(stopwords.words("english"))
stopwords.update(['.', ',', '"', "'", '?', '!',
                  ':', ';', '(', ')', '[', ']', '{', '}'])


def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopwords]


def tokenize(text):
    return wordpunct_tokenize(text)


def preprocess(text):
    return remove_stopwords(tokenize(text))
