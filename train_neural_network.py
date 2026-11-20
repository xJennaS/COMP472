"""Train NumPy MLPs for both tasks and create a combined submission."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from make_submission import build_submission
from src.data_loader import load_task, split_train_val, standardize


_BATCH_SIZE = 256
_MAX_EPOCHS = 40
_PATIENCE = 7
_MODEL_CONFIGS = (
    {"hidden_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4},
    {"hidden_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4},
    {"hidden_size": 128, "learning_rate": 3e-4, "weight_decay": 1e-3},
    {"hidden_size": 256, "learning_rate": 3e-4, "weight_decay": 1e-4},
)


def _initialize(
    input_size: int,
    hidden_size: int,
    output_size: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    return {
        "w1": rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size)).astype(np.float32),
        "b1": np.zeros(hidden_size, dtype=np.float32),
        "w2": rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size)).astype(np.float32),
        "b2": np.zeros(output_size, dtype=np.float32),
    }


def _forward(
    X: np.ndarray, parameters: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    hidden = np.maximum(0, X @ parameters["w1"] + parameters["b1"])
    logits = hidden @ parameters["w2"] + parameters["b2"]
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return hidden, probabilities


def _measure(
    X: np.ndarray, y: np.ndarray, parameters: Dict[str, np.ndarray]
) -> Tuple[float, float]:
    _, probabilities = _forward(X, parameters)
    loss = -np.log(
        np.maximum(probabilities[np.arange(len(y)), y], 1e-12)
    ).mean()
    accuracy = (probabilities.argmax(axis=1) == y).mean()
    return float(loss), float(accuracy)


def _train(
    X: np.ndarray,
    y: np.ndarray,
    output_size: int,
    epochs: int,
    seed: int,
    hidden_size: int,
    learning_rate: float,
    weight_decay: float,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], int]:
    rng = np.random.default_rng(seed)
    parameters = _initialize(X.shape[1], hidden_size, output_size, rng)
    first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    best_parameters = {name: value.copy() for name, value in parameters.items()}
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    step = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))
        for start in range(0, len(y), _BATCH_SIZE):
            indices = order[start : start + _BATCH_SIZE]
            batch_X = X[indices]
            batch_y = y[indices]
            hidden, probabilities = _forward(batch_X, parameters)

            grad_logits = probabilities
            grad_logits[np.arange(len(batch_y)), batch_y] -= 1
            grad_logits /= len(batch_y)
            gradients = {
                "w2": hidden.T @ grad_logits + weight_decay * parameters["w2"],
                "b2": grad_logits.sum(axis=0),
            }
            grad_hidden = grad_logits @ parameters["w2"].T
            grad_hidden[hidden <= 0] = 0
            gradients["w1"] = (
                batch_X.T @ grad_hidden + weight_decay * parameters["w1"]
            )
            gradients["b1"] = grad_hidden.sum(axis=0)

            step += 1
            for name, gradient in gradients.items():
                first_moment[name] = (
                    0.9 * first_moment[name] + 0.1 * gradient
                )
                second_moment[name] = (
                    0.999 * second_moment[name] + 0.001 * gradient**2
                )
                corrected_first = first_moment[name] / (1 - 0.9**step)
                corrected_second = second_moment[name] / (1 - 0.999**step)
                parameters[name] -= learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + 1e-8
                )

        if X_val is None or y_val is None:
            continue

        val_loss, val_accuracy = _measure(X_val, y_val, parameters)
        if val_accuracy > best_accuracy or (
            val_accuracy == best_accuracy and val_loss < best_loss
        ):
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_epoch = epoch
            best_parameters = {
                name: value.copy() for name, value in parameters.items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= _PATIENCE:
            break

    return best_parameters if X_val is not None else parameters, best_epoch or epochs


def train_task(task: str, output_dir: Path, seed: int) -> Tuple[int, float]:
    X, y, X_test, test_ids = load_task(task)
    X_train, X_val, y_train, y_val = split_train_val(X, y, seed=seed)
    X_train, X_val = standardize(X_train, X_val)
    output_size = int(y.max()) + 1

    print(
        f"{task}: training {len(y_train)} rows, validating on {len(y_val)} rows",
        flush=True,
    )
    best_config = None
    best_epoch = 0
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    for config in _MODEL_CONFIGS:
        validation_parameters, config_epoch = _train(
            X_train,
            y_train,
            output_size,
            _MAX_EPOCHS,
            seed,
            config["hidden_size"],
            config["learning_rate"],
            config["weight_decay"],
            X_val,
            y_val,
        )
        val_loss, val_accuracy = _measure(
            X_val, y_val, validation_parameters
        )
        print(
            f"  hidden={config['hidden_size']}, lr={config['learning_rate']}, "
            f"weight_decay={config['weight_decay']}: "
            f"validation accuracy={val_accuracy:.4f}, best epoch={config_epoch}",
            flush=True,
        )
        if val_accuracy > best_validation_accuracy or (
            val_accuracy == best_validation_accuracy
            and val_loss < best_validation_loss
        ):
            best_config = config
            best_epoch = config_epoch
            best_validation_accuracy = val_accuracy
            best_validation_loss = val_loss
    if best_config is None:
        raise RuntimeError(f"No neural-network configuration was evaluated for {task}")

    print(
        f"{task}: selected hidden={best_config['hidden_size']}, "
        f"lr={best_config['learning_rate']}, "
        f"weight_decay={best_config['weight_decay']}",
        flush=True,
    )
    # Refit using all labeled data after selecting the stopping point on validation.
    X_all, _, X_test = standardize(X, X, X_test)
    final_parameters, _ = _train(
        X_all,
        y,
        output_size,
        best_epoch,
        seed + 1,
        best_config["hidden_size"],
        best_config["learning_rate"],
        best_config["weight_decay"],
    )
    _, test_probabilities = _forward(X_test, final_parameters)
    predictions = test_probabilities.argmax(axis=1)

    prediction_path = output_dir / f"{task}_predictions.csv"
    pd.DataFrame({"id": test_ids, "label": predictions}).to_csv(
        prediction_path, index=False
    )
    print(f"{task}: best validation accuracy={best_validation_accuracy:.4f}")
    print(f"{task}: saved test predictions to {prediction_path}", flush=True)
    return best_epoch, best_validation_accuracy


def main() -> None:
    repository_dir = Path(__file__).resolve().parent
    image_epochs, image_accuracy = train_task(
        "image", repository_dir, seed=42
    )
    sentiment_epochs, sentiment_accuracy = train_task(
        "sentiment", repository_dir, seed=43
    )
    print(
        f"Validation results: image={image_accuracy:.4f}, "
        f"sentiment={sentiment_accuracy:.4f}",
        flush=True,
    )
    build_submission(
        repository_dir / "image_predictions.csv",
        repository_dir / "sentiment_predictions.csv",
        repository_dir / "data",
        repository_dir / "submission.csv",
    )


if __name__ == "__main__":
    main()
