"""Data loading and preprocessing utilities for the COMP 472 project."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, overload

import numpy as np
import pandas as pd


_TASK_CONFIG = {
    "image": {"feature_count": 500, "label_min": 0, "label_max": 49},
    "sentiment": {"feature_count": 768, "label_min": 0, "label_max": 2},
}


def _validate_columns(
    frame: pd.DataFrame,
    file_path: Path,
    feature_count: int,
    require_label: bool,
) -> list[str]:
    required = {"id"}
    if require_label:
        required.add("label")
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{file_path} is missing required column(s): "
            f"{', '.join(sorted(missing_columns))}"
        )

    feature_columns = [
        column for column in frame.columns if column.startswith("feature_")
    ]
    if len(feature_columns) != feature_count:
        raise ValueError(
            f"{file_path} must contain exactly {feature_count} feature columns "
            f"(found {len(feature_columns)})"
        )

    expected_features = [f"feature_{index}" for index in range(feature_count)]
    absent_features = [name for name in expected_features if name not in frame.columns]
    if absent_features:
        raise ValueError(
            f"{file_path} is missing expected feature column(s): "
            f"{', '.join(absent_features)}"
        )

    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(
            f"{file_path} contains missing values in column(s): "
            f"{', '.join(missing)}"
        )
    return expected_features


def _read_csv(
    file_path: Path, feature_count: int, require_label: bool
) -> Tuple[pd.DataFrame, list[str]]:
    if not file_path.is_file():
        raise FileNotFoundError(f"Required data file not found: {file_path}")
    frame = pd.read_csv(file_path)
    feature_columns = _validate_columns(
        frame, file_path, feature_count, require_label
    )
    return frame, feature_columns


def _get_features(
    frame: pd.DataFrame, feature_columns: list[str], file_path: Path
) -> np.ndarray:
    try:
        features = frame[feature_columns].to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{file_path} contains non-numeric feature values") from error
    if not np.isfinite(features).all():
        raise ValueError(f"{file_path} contains non-finite feature values")
    return features


def load_task(
    task: str, data_dir: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate one task's train and test CSV files.

    Args:
        task: Either ``"image"`` or ``"sentiment"``.
        data_dir: Directory containing the CSV files. Defaults to the
            repository-root ``data/raw/`` directory.

    Returns:
        ``X_train``, ``y_train``, ``X_test``, and ``test_ids``. Test rows and
        their IDs retain the order in the source CSV.
    """
    if task not in _TASK_CONFIG:
        raise ValueError(
            f"Unknown task {task!r}; expected 'image' or 'sentiment'"
        )

    config = _TASK_CONFIG[task]
    directory = (
        Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "data" / "raw"
    )
    train_path = directory / f"{task}_train.csv"
    test_path = directory / f"{task}_test.csv"

    train_frame, feature_columns = _read_csv(
        train_path, config["feature_count"], require_label=True
    )
    test_frame, test_feature_columns = _read_csv(
        test_path, config["feature_count"], require_label=False
    )

    try:
        numeric_labels = pd.to_numeric(train_frame["label"], errors="raise").to_numpy()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{train_path} contains non-numeric labels") from error
    if not np.isfinite(numeric_labels).all() or not np.equal(
        numeric_labels, np.floor(numeric_labels)
    ).all():
        raise ValueError(f"{train_path} labels must be finite integers")
    if (
        (numeric_labels < config["label_min"]).any()
        or (numeric_labels > config["label_max"]).any()
    ):
        raise ValueError(
            f"{train_path} labels must be in the range "
            f"{config['label_min']}-{config['label_max']}"
        )

    X_train = _get_features(train_frame, feature_columns, train_path)
    X_test = _get_features(test_frame, test_feature_columns, test_path)
    y_train = numeric_labels.astype(int)
    test_ids = test_frame["id"].to_numpy()
    return X_train, y_train, X_test, test_ids


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic stratified train/validation split using NumPy."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array (got {X.ndim} dimensions)")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array (got {y.ndim} dimensions)")
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same number of rows ({len(X)} != {len(y)})")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if len(y) == 0:
        raise ValueError("X and y must contain at least one sample")

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for label in np.unique(y):
        class_indices = rng.permutation(np.flatnonzero(y == label))
        class_size = len(class_indices)
        if class_size < 2:
            raise ValueError(
                f"Each class needs at least 2 samples for a stratified split "
                f"(class {label!r} has {class_size})"
            )
        val_size = min(class_size - 1, max(1, int(round(class_size * val_fraction))))
        val_indices.extend(class_indices[:val_size])
        train_indices.extend(class_indices[val_size:])

    train_indices = rng.permutation(train_indices)
    val_indices = rng.permutation(val_indices)
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]


@overload
def standardize(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_test: None = None,
) -> Tuple[np.ndarray, np.ndarray]: ...


@overload
def standardize(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def standardize(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Standardize features using only the training split's mean and std.

    Returns two arrays when ``X_test`` is omitted, or three when it is given.
    Features with zero training standard deviation are left unscaled.
    """
    train = np.asarray(X_tr, dtype=np.float32)
    validation = np.asarray(X_val, dtype=np.float32)
    test = None if X_test is None else np.asarray(X_test, dtype=np.float32)
    arrays = [("X_tr", train), ("X_val", validation)]
    if test is not None:
        arrays.append(("X_test", test))
    for name, array in arrays:
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D array (got {array.ndim} dimensions)")
        if array.shape[1] != train.shape[1]:
            raise ValueError(
                f"{name} has {array.shape[1]} features; expected {train.shape[1]}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")

    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    standardized = [
        ((array - mean) / std).astype(np.float32, copy=False)
        for _, array in arrays
    ]
    if test is None:
        return standardized[0], standardized[1]
    return standardized[0], standardized[1], standardized[2]


if __name__ == "__main__":
    for task_name in _TASK_CONFIG:
        X_train, y_train, X_test, test_ids = load_task(task_name)
        classes, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(classes.tolist(), counts.tolist()))
        print(
            f"{task_name}: X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_test={X_test.shape}, test_ids={test_ids.shape}, "
            f"class_counts={class_counts}"
        )
