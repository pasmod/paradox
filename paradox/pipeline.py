from sklearn.pipeline import Pipeline, FeatureUnion


def pipeline(transformers=None, estimator=None):
    return Pipeline(steps=[('union', FeatureUnion(transformers)), estimator])
