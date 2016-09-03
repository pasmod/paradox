from sklearn.feature_extraction.text import CountVectorizer

def tokenize(text):
    return text.split(" ")


def get_vocabulary(X_tuples):
    a = [(x[0]) for x in X_tuples]
    b = [(x[1]) for x in X_tuples]
    vectorizer = CountVectorizer(tokenizer=tokenize)
    vectorizer.fit(a)
    a_names = set(vectorizer.get_feature_names())
    vectorizer = CountVectorizer(tokenizer=tokenize)
    vectorizer.fit(b)
    b_names = set(vectorizer.get_feature_names())
    return list(a_names | b_names)
