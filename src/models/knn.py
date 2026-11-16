from collections import Counter

import numpy as np


def euclidean_distance(a, b):
    """Compute straight-line distance, summing over the feature dimension."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def manhattan_distance(a, b):
    """Compute the sum of absolute feature differences."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.abs(a - b), axis=-1)


def minkowski_distance(a, b, p):
    """Compute the Lp distance over the feature dimension."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.abs(a - b) ** p, axis=-1) ** (1 / p)


class KNNClassifier:
    def __init__(self, k=3, distance_fn=euclidean_distance, voting_method="majority"):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        if voting_method not in ("majority", "weighted"):
            raise ValueError("voting_method must be 'majority' or 'weighted'")
        self.k = k
        self.distance_fn = distance_fn
        self.voting_method = voting_method
        self.X_train = []
        self.y_train = []

    def fit(self, X_train, y_train):
        """k-NN has no training step; we simply store the data."""
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if X_train.ndim != 2 or y_train.ndim != 1:
            raise ValueError("X_train must be 2D and y_train must be 1D")
        if len(X_train) != len(y_train):
            raise ValueError("Training features and labels must have the same length")
        if self.k > len(X_train):
            raise ValueError("k must be between 1 and the number of training samples")
        self.X_train = X_train
        self.y_train = y_train
        return self

    def _predict_one(self, x):
        # Calculate distances.
        distances = self.distance_fn(x, self.X_train)
        dists = sorted(
            [(distance, yi) for distance, yi in zip(distances, self.y_train)],
            key=lambda d: d[0],
        )
        neighbours = dists[:self.k]
        if self.voting_method == "majority":
            k_labels = [label for _, label in neighbours]
            return Counter(k_labels).most_common(1)[0][0]

        scores = {}
        for distance, label in neighbours:
            weight = 1 / (distance + 1e-10)
            scores[label] = scores.get(label, 0) + weight
        return max(scores, key=scores.get)

    def predict(self, X_test):
        if len(self.X_train) == 0:
            raise ValueError("Call fit before predict")
        X_test = np.asarray(X_test)
        if X_test.ndim != 2 or X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError("X_test must be 2D with the same number of features as X_train")
        return [self._predict_one(x) for x in X_test]

    def score(self, X_test, y_test):
        if len(y_test) == 0 or len(X_test) != len(y_test):
            raise ValueError("Test features and labels must have the same nonzero length")
        preds = self.predict(X_test)
        return sum(p == t for p, t in zip(preds, y_test)) / len(y_test)
