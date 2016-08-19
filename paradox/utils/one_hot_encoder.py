from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def get_one_hot_encoding(X, vocabulary, tokenizer=None, analyzer='word'):
    sentences_A = [(x[0]) for x in X]
    sentences_B = [(x[1]) for x in X]
    vectorizer = CountVectorizer(analyzer=analyzer, tokenizer=tokenizer, vocabulary=vocabulary, binary=True)
    encodings_sentences_A = vectorizer.transform(sentences_A).toarray()
    encodings_sentences_B = vectorizer.transform(sentences_B).toarray()
    return np.concatenate((encodings_sentences_A, encodings_sentences_B), axis=1)
