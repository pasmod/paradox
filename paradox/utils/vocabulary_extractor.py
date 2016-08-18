from sklearn.feature_extraction.text import CountVectorizer


def get_vocabulary(X_tuples, analyzer='word'):
    X = [(' '.join(x)) for x in X_tuples]
    vectorizer = CountVectorizer(analyzer=analyzer)
    vectorizer.fit(X)
    return vectorizer.get_feature_names()
