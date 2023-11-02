import pandas as pd
import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    if y_true.unique().shape[0] > 2:
        raise ValueError("Precision is only defined for binary classification")
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp+fp)

def recall(y_true, y_pred):
    if y_true.unique().shape[0] > 2:
        raise ValueError("Recall is only defined for binary classification")
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp+fn)

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r)/(p+r)

def roc_auc(y_true, y_pred):
    pass