import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=-1))


def initialize_centroids(X, k, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


def assign_clusters(X, centroids):
    distances = np.empty((len(X), len(centroids)))
    for j, centroid in enumerate(centroids):
        distances[:, j] = euclidean_distance(X, centroid)
    return np.argmin(distances, axis=1)


def update_centroids(X, assignments, k, rng):
    centroids = np.empty((k, X.shape[1]))
    for j in range(k):
        points = X[assignments == j]
        if len(points) > 0:
            centroids[j] = np.mean(points, axis=0)
        else:
            centroids[j] = X[rng.integers(len(X))]
    return centroids


def compute_inertia(X, assignments, centroids):
    return float(np.sum((X - centroids[assignments]) ** 2))


class KMeans:
    def __init__(self, k=3, max_iter=100, tol=1e-4, seed=42):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and nonnegative")
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty 2D numeric array")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if self.k > len(X):
            raise ValueError("k must not exceed the number of samples")

        rng = np.random.default_rng(self.seed)
        self.centroids_ = initialize_centroids(X, self.k, self.seed)
        self.n_iter_ = 0

        for _ in range(self.max_iter):
            assignments = assign_clusters(X, self.centroids_)
            new_centroids = update_centroids(X, assignments, self.k, rng)
            shift = max(euclidean_distance(self.centroids_, new_centroids))
            self.centroids_ = new_centroids
            self.n_iter_ += 1
            if shift < self.tol:
                break

        self.labels_ = assign_clusters(X, self.centroids_)
        self.inertia_ = compute_inertia(X, self.labels_, self.centroids_)
        return self

    def predict(self, X):
        if self.centroids_ is None:
            raise ValueError("Call fit before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.centroids_.shape[1]:
            raise ValueError("X must be 2D with the same number of features as the centroids")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return assign_clusters(X, self.centroids_)
