from scipy.spatial.distance import cosine
from pythonrouge import pythonrouge
from preprocessor import preprocess
from glove import Glove
import numpy as np

glove = Glove.load(dim=200)


def surface(text1, text2, method='ROUGE-2'):
    methods = ['ROUGE-1', 'ROUGE-2', 'ROUGE-3', 'ROUGE-SU4', 'ROUGE-L']
    if method not in methods:
        raise ValueError("Method {} is not supported."
                         "Available methods: {}".format(method, methods))
    return pythonrouge.pythonrouge(text1, text2)[method]


def context(text1, text2):
    tokens1 = preprocess(text1)
    print tokens1
    tokens2 = preprocess(text2)
    print tokens2
    vectors1 = filter(lambda v: v, [glove.vector(token) for token in tokens1])
    print len(vectors1)
    vectors2 = filter(lambda v: v, [glove.vector(token) for token in tokens2])
    print vectors2
    center1 = np.average(np.array(vectors1), axis=0)
    print len(vectors2)
    print center1
    center2 = np.average(np.array(vectors2), axis=0)
    print center2
    return 1 - cosine(center1, center2)
