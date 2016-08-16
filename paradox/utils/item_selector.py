from sklearn.base import BaseEstimator, TransformerMixin


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dimension):
        self.dimension = dimension

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        projection = [(x[self.dimension]) for x in X]
        return projection
