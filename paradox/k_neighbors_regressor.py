from sklearn import neighbors
from sklearn.pipeline import Pipeline


def param_grid():
    return {'estimator__kneighbors_regressor__n_neighbors': [1, 3, 5, 7]}


def build(n_neighbors=1):
    pipeline = Pipeline([(
        'kneighbors_regressor',
        neighbors.KNeighborsRegressor(n_neighbors=n_neighbors))])
    return ('estimator', pipeline)
