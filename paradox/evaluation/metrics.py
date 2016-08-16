from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from pandas_confusion import ConfusionMatrix
import numpy as np


def calculate_and_print_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print "*** Results ***" + '*' * 5
    print "Accuracy:" + str(accuracy)
    f1 = f1_score(y_true, y_pred, average=None)
    print "Macro F1:" + str(f1)
    print "Average F1:" + str(np.mean(f1))
    print "Confusion matrix"
    print(ConfusionMatrix(y_true, y_pred))