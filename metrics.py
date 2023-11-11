import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp+fp) if (tp+fp) != 0 else 0

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp+fn) if (tp+fn) != 0 else 0

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p+r) if p+r != 0 else 0

def roc_auc(y_true, y_pred, plot=False):
    thresholds = np.linspace(0, 1, 100)
    tpr = []
    fpr = []
    for threshold in thresholds:
        classifications = np.where(y_pred >= threshold, 1, 0)
        tp = np.sum((classifications == 1) & (y_true == 1))
        fp = np.sum((classifications == 1) & (y_true == 0))
        tn = np.sum((classifications == 0) & (y_true == 0))
        fn = np.sum((classifications == 0) & (y_true == 1))
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    if plot:
        plt.plot(fpr, tpr)
    return abs(np.trapz(fpr, tpr))