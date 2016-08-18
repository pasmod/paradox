from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from utils.item_selector import ItemSelector
from tokenizers.hindi_tokenizer_wrapper import hindi_tokenize


def create_count_vectorizer_pipeline(use_hindi_tokenizer=True):
    tokenizer = None
    if use_hindi_tokenizer:
        tokenizer = hindi_tokenize
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
