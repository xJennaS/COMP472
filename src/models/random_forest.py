# Adapted from COMP472 Lab 5, cells 40 and 43.
import math
import random
from collections import Counter

from src.models.decision_tree import DecisionTreeClassifier, predict_one


def bootstrap_sample(X, y, seed=None):
    """Draw n samples with replacement. Returns (X_boot, y_boot, oob_indices)."""
    if seed is not None:
        random.seed(seed)
    n = len(X)
    idx = [random.randint(0, n - 1) for _ in range(n)]
    oob = sorted(set(range(n)) - set(idx))
    X_boot = [X[i] for i in idx]
    y_boot = [y[i] for i in idx]
    return X_boot, y_boot, oob

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples=1,
                 n_features='sqrt', seed=42):
        self.n_trees     = n_trees
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.n_features  = n_features   # 'sqrt', 'log2', or int
        self.seed        = seed
        self.trees       = []
        self._oob_preds  = {}           # sample index -> list of OOB predictions

    def _get_k(self, n_feats):
        if self.n_features == 'sqrt':
            return max(1, int(math.sqrt(n_feats)))
        if self.n_features == 'log2':
            return max(1, int(math.log2(n_feats)))
        return self.n_features

    def fit(self, X, y):
        n_feats = len(X[0])
        k = self._get_k(n_feats)
        self.trees = []
        self._oob_preds = {i: [] for i in range(len(X))}

        for t in range(self.n_trees):
            X_boot, y_boot, oob_idx = bootstrap_sample(X, y, seed=self.seed + t)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                n_features=k
            ).fit(X_boot, y_boot)
            self.trees.append(tree)

            # Collect OOB predictions for error estimation
            for i in oob_idx:
                self._oob_preds[i].append(predict_one(tree.root, X[i]))

        return self

    def predict(self, X):
        # Majority vote across all trees
        all_votes = [[predict_one(t.root, x) for t in self.trees] for x in X]
        return [Counter(votes).most_common(1)[0][0] for votes in all_votes]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(p == t for p, t in zip(preds, y)) / len(y)

    def oob_score(self, y):
        """Accuracy estimated using only out-of-bag predictions (no held-out set needed)."""
        correct = total = 0
        for i, preds in self._oob_preds.items():
            if preds:
                oob_pred = Counter(preds).most_common(1)[0][0]
                correct += (oob_pred == y[i])
                total   += 1
        return correct / total if total > 0 else 0.0
