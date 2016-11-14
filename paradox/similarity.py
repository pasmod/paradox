from scipy.spatial.distance import cosine
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
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
    tokens2 = preprocess(text2)
    vectors1 = filter(lambda v: v, [glove.vector(token) for token in tokens1])
    vectors2 = filter(lambda v: v, [glove.vector(token) for token in tokens2])
    center1 = np.average(np.array(vectors1), axis=0)
    center2 = np.average(np.array(vectors2), axis=0)
    return 1 - cosine(center1, center2)


def similarity(text1, text2, levels=['surface', 'context']):
    sims = []
    for level in levels:
        if level == 'surface':
            sims.append(surface(text1, text2))
        if level == 'context':
            sims.append(context(text1, text2))
        else:
            raise ValueError("Level {} not supported!".format(level))
    return sims


def build(levels=['surface', 'context']):
    pipeline = Pipeline([('transformer', Similarity(levels=levels))])
    return ('similarity', pipeline)


def param_grid():
    return {"union__similarity__transformer__levels":
            [["surface"], ["context"], ["surface", "context"]]}


class Similarity(BaseEstimator):
    def __init__(self, levels=['surface']):
        self.levels = levels

    def fit(self, X, y):
        return X

    def transform(self, X):
        return map(lambda x: self._transform(x), X)

    def _transform(self, x):
        pair = x.split("<<STOP>>")
        return similarity(pair[0], pair[1], self.levels)
