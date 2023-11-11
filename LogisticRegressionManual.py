import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionManual:

    def __init__(self, num=500, eta=0.001):
        self.num = num
        self.eta = eta

    def fit(self, X, y):
        X = pd.get_dummies(X).astype(int)
        xCols = X.columns
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate([ones, X], axis=1)
        target = y.name
        c = y.values
        self.betas = np.zeros(X.shape[1])
        for i in range(self.num):
            zi = X @ self.betas
            p = 1 / (1 + np.exp(-zi))
            # feel free to play with the loss function. This is used because it doesn't fail
            dl = X.T @ (2 * (c - p) * -p ** 2 * -np.exp(-zi)) + 15 * self.betas
            if np.sum(np.sign(self.betas) != np.sign(self.betas - self.eta * dl)) > 2:
                self.eta /= 2
            self.betas = self.betas - self.eta * dl
            loss = np.sum(dl ** 2)
            if loss < 1e-4:
                break
        self.betas = pd.DataFrame(self.betas, index=["Intercept"] + xCols.tolist(), columns=["ß"])
        return self.betas

    def predict_proba(self, X_test):
        X_test = pd.get_dummies(X_test).astype(int)
        ones = np.ones((X_test.shape[0], 1))
        X_test = np.concatenate([ones, X_test], axis=1)
        zi = X_test @ self.betas
        p = 1 / (1 + np.exp(-zi))
        stuff = pd.DataFrame([1-p, p], columns=[0, 1])
        return stuff

    def predict(self, X_test, threshold=0.5):
        probas = self.predict_proba(X_test)
        return np.where(probas[1] >= threshold, 1, 0)