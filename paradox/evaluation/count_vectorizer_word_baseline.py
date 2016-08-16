from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from metrics import calculate_and_print_metrics


def estimate_svm_baseline(test_train_split):
    pipeline = create_pipeline(test_train_split)
    pipeline.fit(test_train_split['X_train'], test_train_split['y_train'])
    predicted_classes = pipeline.predict(test_train_split['X_test'])
    calculate_and_print_metrics(test_train_split['y_test'], predicted_classes)


def create_pipeline(test_train_split):
    return Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('sentence_a', Pipeline([
                    ('selector', ItemSelector(dimension=0)),
                    ('vect', CountVectorizer()),
                ])),
                ('sentence_b', Pipeline([
                    ('selector', ItemSelector(dimension=1)),
                    ('vect', CountVectorizer()),
                ])),
            ],
        )),
        ('clf', SVC()),
    ])


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dimension):
        self.dimension = dimension

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        projection = [(x[self.dimension]) for x in X]
        return projection
