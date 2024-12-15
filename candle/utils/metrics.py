import numpy as np
from sklearn.metrics import roc_auc_score


def accuracy(true, pred):
    true = true.argmax(axis=1)
    pred = pred.argmax(axis=1)
    return (true == pred).sum() / len(pred)


def auc(true, pred):
    return roc_auc_score(true, pred)