from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from metrics import calculate_and_print_metrics
from sklearn.grid_search import GridSearchCV
from time import time
from utils.item_selector import ItemSelector
from tokenizers.hindi_tokenizer import Tokenizer


def estimate_svm_baseline(test_train_split):
    t0 = time()
    use_hindi_tokenizer = True
    pipeline = create_pipeline(test_train_split, use_hindi_tokenizer)

    param_grid = {'svm__C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'svm__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
    print("Starting grid search")
    grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=3)
    grid_search.fit(test_train_split['X_train'], test_train_split['y_train'])
    predicted_classes = grid_search.predict(test_train_split['X_test'])
    # classification
    # pipeline.fit(test_train_split['X_train'], test_train_split['y_train'])
    # predicted_classes = pipeline.predict(test_train_split['X_test'])
    calculate_and_print_metrics(test_train_split['y_test'], predicted_classes)
    print("Total execution time in %0.3fs" % (time() - t0))
    print '*' * 20
    print ''


def hindi_tokenizer(s):
    tokenizer = Tokenizer(s)
    tokenizer.tokenize()
    return tokenizer.tokens


def create_pipeline(test_train_split, use_hindi_tokenizer=True):
    tokenizer = None
    if use_hindi_tokenizer:
        tokenizer = hindi_tokenizer
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
