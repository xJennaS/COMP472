import math
import random
from collections import Counter


def entropy(y):
    n = len(y)
    if n == 0:
        return 0.0
    counts = Counter(y)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def information_gain(y_parent, y_left, y_right):
    n = len(y_parent)
    if n == 0:
        return 0.0
    w_left = len(y_left) / n
    w_right = len(y_right) / n
    return entropy(y_parent) - (w_left * entropy(y_left) + w_right * entropy(y_right))


def gini(y):
    n = len(y)
    if n == 0:
        return 0.0
    counts = Counter(y)
    return 1 - sum((c / n) ** 2 for c in counts.values())


def gini_gain(y_parent, y_left, y_right):
    n = len(y_parent)
    if n == 0:
        return 0.0
    w_left = len(y_left) / n
    w_right = len(y_right) / n
    return gini(y_parent) - (w_left * gini(y_left) + w_right * gini(y_right))


class TreeNode:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = None

    def is_leaf(self):
        return self.label is not None


def find_best_split(X, y, n_features=None, criterion="entropy"):
    n_samples = len(X)
    n_feats = len(X[0])
    feat_indices = (
        random.sample(range(n_feats), min(n_features, n_feats))
        if n_features is not None
        else range(n_feats)
    )
    best_gain, best_feat, best_thresh = -1, None, None

    for fi in feat_indices:
        vals = sorted(set(row[fi] for row in X))
        thresholds = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
        for thresh in thresholds:
            y_left = [y[i] for i in range(n_samples) if X[i][fi] <= thresh]
            y_right = [y[i] for i in range(n_samples) if X[i][fi] > thresh]
            if not y_left or not y_right:
                continue
            if criterion == "gini":
                gain = gini_gain(y, y_left, y_right)
            else:
                gain = information_gain(y, y_left, y_right)
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, fi, thresh

    return best_feat, best_thresh, best_gain


def build_tree(X, y, max_depth=None, min_samples=1, depth=0, n_features=None, criterion="entropy"):
    node = TreeNode()
    if (len(set(y)) == 1
            or len(y) <= min_samples
            or (max_depth is not None and depth >= max_depth)):
        node.label = Counter(y).most_common(1)[0][0]
        return node

    feat_idx, thresh, gain = find_best_split(X, y, n_features=n_features, criterion=criterion)
    if feat_idx is None or gain <= 0:
        node.label = Counter(y).most_common(1)[0][0]
        return node

    node.feature_idx = feat_idx
    node.threshold = thresh
    left_mask = [X[i][feat_idx] <= thresh for i in range(len(X))]
    X_left = [X[i] for i in range(len(X)) if left_mask[i]]
    y_left = [y[i] for i in range(len(y)) if left_mask[i]]
    X_right = [X[i] for i in range(len(X)) if not left_mask[i]]
    y_right = [y[i] for i in range(len(y)) if not left_mask[i]]

    node.left = build_tree(X_left, y_left, max_depth, min_samples, depth + 1, n_features, criterion)
    node.right = build_tree(X_right, y_right, max_depth, min_samples, depth + 1, n_features, criterion)
    return node


def predict_one(tree, x):
    if tree.is_leaf():
        return tree.label
    if x[tree.feature_idx] <= tree.threshold:
        return predict_one(tree.left, x)
    else:
        return predict_one(tree.right, x)


def predict(tree, X):
    return [predict_one(tree, x) for x in X]


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples=1, n_features=None, criterion="entropy"):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        X = [[float(value) for value in row] for row in X]
        y = list(y)
        self.root = build_tree(X, y, self.max_depth, self.min_samples,
                               n_features=self.n_features, criterion=self.criterion)
        return self

    def predict(self, X):
        return predict(self.root, X)

    def score(self, X, y):
        preds = self.predict(X)
        return sum(p == t for p, t in zip(preds, y)) / len(y)
