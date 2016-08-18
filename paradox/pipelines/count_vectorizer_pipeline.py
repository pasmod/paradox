from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from utils.item_selector import ItemSelector


def create_count_vectorizer_pipeline(tokenizer=None):
    return Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('sentence_a', Pipeline([
                    ('selector', ItemSelector(dimension=0)),
                    ('vect', CountVectorizer(tokenizer=tokenizer)),
                ])),
                ('sentence_b', Pipeline([
                    ('selector', ItemSelector(dimension=1)),
                    ('vect', CountVectorizer(tokenizer=tokenizer)),
                ])),
            ],
        )),
        ('svm', SVC()),
    ])


def create_char_count_vectorizer_pipeline():
    return Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('sentence_a', Pipeline([
                    ('selector', ItemSelector(dimension=0)),
                    ('vect', CountVectorizer(analyzer='char_wb')),
                ])),
                ('sentence_b', Pipeline([
                    ('selector', ItemSelector(dimension=1)),
                    ('vect', CountVectorizer(analyzer='char_wb')),
                ])),
            ],
        )),
        ('svm', SVC()),
    ])
