from sklearn.feature_extraction.text import CountVectorizer


def get_vocabulary(X_tuples, analyzer='word', tokenizer=None, ngram_range=(1, 1)):
    a = [(x[0]) for x in X_tuples]
    b = [(x[1]) for x in X_tuples]
    vectorizer = CountVectorizer(analyzer=analyzer, tokenizer=tokenizer, ngram_range=ngram_range)
    vectorizer.fit(a)
    a_names = set(vectorizer.get_feature_names())
    vectorizer = CountVectorizer(analyzer=analyzer, tokenizer=tokenizer, ngram_range=ngram_range)
    vectorizer.fit(b)
    b_names = set(vectorizer.get_feature_names())
    return list(a_names | b_names)
