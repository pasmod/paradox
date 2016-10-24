def ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])


def tokenize(text):
    return text.split(" ")
