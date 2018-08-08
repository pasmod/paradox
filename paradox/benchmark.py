from metrics import pearson, mse
from pipeline import pipeline
import k_neighbors_regressor
import numpy as np
import similarity
import parser


def report(correlations, errors, y_pred_fold):
    print("PC:\t\t\t%0.2f\t(+/- %0.2f)" % (np.mean(correlations),
                                           np.std(correlations) * 2))
    print("RMSE:\t\t\t%0.2f\t(+/- %0.2f)" % (np.mean(errors),
                                             np.std(errors) * 2))
    print('{}'.format('*' * 40))


def test(model=None, categories=[]):
    print("Testing on Category {}".format(categories))
    test_pairs = parser.parse(mode="test", categories=categories)
    X = [pair[0] for pair in test_pairs]
    y = [pair[1] for pair in test_pairs]
    y_pred = p.predict(X)
    pcs = [pearson(y, y_pred)]
    rmses = [mse(y, y_pred)]
    report(pcs, rmses, y_pred)


pairs = parser.parse(mode="train")
X = [pair[0] for pair in pairs]
y = [pair[1] for pair in pairs]
transformer = similarity.build()
estimator = k_neighbors_regressor.build(n_neighbors=4)
p = pipeline(transformers=[transformer], estimator=estimator)
p.fit(X, y)


test(p, categories=["answer-answer"])
test(p, categories=["question-question"])
test(p, categories=["headlines"])
test(p, categories=["postediting"])
test(p, categories=["plagiarism"])
