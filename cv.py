import pandas as pd
import numpy as np
from metrics import *
import warnings
warnings.filterwarnings("ignore")

def dummies(data):
    dummyCols = [col for col in data.columns if data[col].nunique() < data.shape[0]] # filter out id columns automatically
    return pd.get_dummies(data[dummyCols]).astype(int)

def splits(data, mode="random", v=5):
    data.reset_index(drop=True, inplace=True)
    data = dummies(data)
    X = data.drop("TARGET", axis=1)
    y = data["TARGET"]
    folds = []
    if mode == "random":
        for i in range(v):
            trainIndex = np.random.choice(X.index, size=int(len(X)*0.8), replace=False)
            testIndex = X.index.difference(trainIndex)
            X_train = X.iloc[trainIndex]
            X_test = X.iloc[testIndex]
            y_train = y.iloc[trainIndex]
            y_test = y.iloc[testIndex]
            folds.append((X_train, X_test, y_train, y_test))
    elif mode == "stratified":
        classes = y.unique()
        classIndices = [y[y==c].index for c in classes]
        classIndices = pd.Series(classIndices)
        for i in range(v):
            choices = classIndices.apply(lambda c: np.random.choice(c, size=int(len(c)*0.8), replace=False)) # build the training set
            trainIndex = np.concatenate(choices)
            testIndex = X.index.difference(trainIndex)
            X_train = X.iloc[trainIndex]
            X_test = X.iloc[testIndex]
            y_train = y.iloc[trainIndex]
            y_test = y.iloc[testIndex]
            folds.append((X_train, X_test, y_train, y_test))
    elif mode == "non-random":
        interval = len(X) // v
        for i in range(v):
            testIndex = np.arange(i * interval, (i+1) * interval)
            testIndex = testIndex[testIndex < len(X)]
            trainIndex = X.index.difference(testIndex)
            trainIndex = trainIndex[trainIndex < len(X)]
            X_train = X.iloc[trainIndex]
            X_test = X.iloc[testIndex]
            y_train = y.iloc[trainIndex]
            y_test = y.iloc[testIndex]
            folds.append((X_train, X_test, y_train, y_test))
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return folds

def cv(data, models, mode="random", v=5):
    folds = splits(data, mode, v)
    things = pd.DataFrame(columns=["model", "fold", "accuracy", "precision", "recall", "f1", "roc_auc"])
    for m in models:
        i = 1
        for X_train, X_test, y_train, y_test in folds:
            try:
                m.probability = True
            except:
                pass
            m.fit(X_train, y_train)
            predictions = m.predict(X_test)
            predictions = pd.Series(predictions, index=X_test.index)
            probabilities = m.predict_proba(X_test)
            probabilities = pd.Series(probabilities[:,1], index=X_test.index)
            stuff = {}
            stuff["accuracy"] = accuracy(y_test, predictions)
            stuff["precision"] = precision(y_test, predictions)
            stuff["recall"] = recall(y_test, predictions)
            stuff["f1"] = f1(y_test, predictions)
            stuff["roc_auc"] = roc_auc(y_test, probabilities)
            stuff["model"] = str(m)
            stuff["fold"] = i
            i += 1
            stuff = pd.DataFrame([stuff])
            things = pd.concat([things, stuff])
    return things