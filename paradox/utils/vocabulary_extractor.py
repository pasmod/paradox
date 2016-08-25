from sklearn.feature_extraction.text import CountVectorizer


def get_vocabulary(X_tuples, analyzer='word', tokenizer=None, ngram_range=(1, 1)):
    X = [(' '.join(x)) for x in X_tuples]
    vectorizer = CountVectorizer(analyzer=analyzer, tokenizer=tokenizer, ngram_range=ngram_range)
    vectorizer.fit(X)
    return vectorizer.get_feature_names()
