from scipy import stats
from sklearn.metrics import mean_squared_error


def pearson(y_truth, y_predicted):
    return stats.pearsonr(y_truth, y_predicted)[0]


def mse(y_truth, y_predicted):
    return mean_squared_error(y_truth, y_predicted)
