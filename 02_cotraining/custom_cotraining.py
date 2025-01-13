import numpy as np
import random
import copy
from sklearn.base import ClassifierMixin, BaseEstimator


class CustomCotrainingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X1, X2, y):
        pass

    def predict(self, X1, X2):
        pass