# src/train.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd

class Trainer:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=500),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

    def train_test_split_data(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_models(self):
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred)
            }
        return results
