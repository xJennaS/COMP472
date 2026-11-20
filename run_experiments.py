"""Run the four PyTorch neural-network experiments for image and sentiment."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data_loader import load_task, split_train_val, standardize
from src.neural_net import (
    ModelConfig,
    TrainingResult,
    config_key,
    one_factor_candidates,
    parameter_count,
    predict_probabilities,
    train_model,
)


SEED = 42
CLASS_NAMES = {
    "image": [str(value) for value in range(50)],
    "sentiment": ["Negative", "Neutral", "Positive"],
}
TASK_LABEL_RANGES = {"image": (0, 49), "sentiment": (0, 2)}
VERSION_CHANGE = {
    1: "Baseline 256-64 MLP with momentum SGD; no regularization.",
    2: "Adam with a one-factor learning-rate and batch-size sweep.",
    3: "Added dropout and weight decay; selected using validation loss and patience.",
    4: "Compared hidden-layer widths and numbers of hidden layers.",
}


def _prepare_directories(root: Path) -> None:
    for task in ("image", "sentiment"):
        (root / "results" / task).mkdir(parents=True, exist_ok=True)
    (root / "results" / "nn").mkdir(parents=True, exist_ok=True)
    (root / "submission" / "nn").mkdir(parents=True, exist_ok=True)


def _plot_sweeps(
    records: list[dict[str, object]], version: int, task: str, path: Path
) -> None:
    if version == 1 or not records:
        return
    factors = list(dict.fromkeys(str(row["factor"]) for row in records))
    figure, axes = plt.subplots(
        1, len(factors), figsize=(5 * len(factors), 4), squeeze=False
    )
    for axis, factor in zip(axes[0], factors):
        selected = [row for row in records if row["factor"] == factor]
        x_values = [str(row["value"]) for row in selected]
        y_values = [float(row["best_val_accuracy"]) for row in selected]
        axis.plot(x_values, y_values, marker="o")
        axis.set(title=factor.replace("_", " ").title(), xlabel=factor, ylabel="Validation accuracy")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.25)
    figure.suptitle(f"v{version} {task}: validation accuracy by hyperparameter")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_version_accuracies(root: Path, summaries: pd.DataFrame) -> None:
    output_dir = root / "results" / "nn"
    for task in ("image", "sentiment"):
        rows = summaries.loc[summaries["task"] == task].sort_values("version")
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(rows["version"], rows["best_val_accuracy"], marker="o")
        axis.set(
            title=f"{task.title()} validation accuracy by version",
            xlabel="Version",
            ylabel="Validation accuracy",
            xticks=[1, 2, 3, 4],
        )
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / f"nn_{task}_accuracy.png", dpi=150)
        plt.close(figure)


def _choose(
    configs: list[tuple[str, str, ModelConfig]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
    version: int,
) -> tuple[ModelConfig, TrainingResult, list[dict[str, object]], float]:
    evaluated: dict[tuple[object, ...], tuple[TrainingResult, float]] = {}
    sweep_rows: list[dict[str, object]] = []
    total_training_time = 0.0
    for factor, value, config in configs:
        key = config_key(config)
        if key not in evaluated:
            started = time.perf_counter()
            result = train_model(
                X_train,
                y_train,
                output_size,
                config,
                epochs,
                seed,
                device,
                X_val,
                y_val,
                early_stopping=(version == 3),
            )
            elapsed = time.perf_counter() - started
            evaluated[key] = (result, elapsed)
            total_training_time += elapsed
        result, _ = evaluated[key]
        sweep_rows.append(
            {
                "version": version,
                "factor": factor,
                "value": value,
                "config": config.description(),
                "best_epoch": result.best_epoch,
                "best_val_accuracy": result.best_accuracy,
                "best_val_loss": result.best_loss,
            }
        )

    winners = [
        (config, *evaluated[config_key(config)])
        for _, _, config in configs
    ]
    winner_config, winner_result, _ = max(
        winners,
        key=lambda item: (item[1].best_accuracy, -item[1].best_loss),
    )
    return winner_config, winner_result, sweep_rows, total_training_time


def _fit_final(
    config: ModelConfig,
    X: np.ndarray,
    y: np.ndarray,
    output_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> TrainingResult:
    return train_model(X, y, output_size, config, epochs, seed, device)


def _save_validation(
    root: Path,
    version: int,
    task: str,
    val_labels: np.ndarray,
    val_predictions: np.ndarray,
) -> pd.DataFrame:
    validation = pd.DataFrame(
        {
            "validation_index": np.arange(len(val_labels)),
            "actual_label": val_labels.astype(int),
            "predicted_label": val_predictions.astype(int),
        }
    )
    validation.to_csv(
        root / "results" / task / f"nn_v{version}_validation.csv", index=False
    )
    return validation


def _build_version_submission(
    root: Path,
    version: int,
    task_predictions: dict[str, pd.DataFrame],
) -> None:
    sample_path = root / "data" / "sample_submission.csv"
    sample = pd.read_csv(sample_path, dtype={"id": "string"})
    if list(sample.columns) != ["id", "label"]:
        raise ValueError(f"{sample_path} must have exactly the columns id,label")
    predictions: list[pd.DataFrame] = []
    for task in ("image", "sentiment"):
        if task not in task_predictions:
            raise ValueError(f"Missing test predictions for {task} v{version}")
        frame = task_predictions[task].copy()
        if list(frame.columns) != ["id", "label"]:
            raise ValueError(
                f"Test predictions for {task} v{version} must have columns id,label"
            )
        labels = pd.to_numeric(frame["label"], errors="raise")
        low, high = TASK_LABEL_RANGES[task]
        if labels.isna().any() or not labels.mod(1).eq(0).all():
            raise ValueError(f"{task} v{version} has missing or non-integer labels")
        if not labels.between(low, high).all():
            raise ValueError(f"{task} v{version} labels must be in {low}-{high}")
        frame["label"] = labels.astype(int)
        predictions.append(frame)
    combined_predictions = pd.concat(predictions, ignore_index=True)
    submission = sample[["id"]].merge(
        combined_predictions, on="id", how="left", validate="one_to_one"
    )
    if len(submission) != 10_000:
        raise ValueError(f"Submission must have 10,000 rows, got {len(submission)}")
    if submission["label"].isna().any():
        raise ValueError("Submission has missing labels after merging test predictions")
    submission["label"] = submission["label"].astype(int)
    if list(submission.columns) != ["id", "label"]:
        raise AssertionError("Submission columns must be exactly id,label")
    output_path = root / "submission" / "nn" / f"submission_v{version}_nn.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)


def _print_summary(
    version: int, task: str, val_accuracy: float, train_accuracy: float,
    previous_accuracy: Optional[float],
) -> None:
    gap = train_accuracy - val_accuracy
    if previous_accuracy is None:
        comparison = "There is no earlier version for comparison."
    else:
        difference = val_accuracy - previous_accuracy
        direction = "up" if difference > 0 else "down" if difference < 0 else "unchanged"
        comparison = f"Validation accuracy went {direction} by {abs(difference):.4f}."
    chance = 1.0 / len(CLASS_NAMES[task])
    if gap >= 0.10:
        fit = "The large train-validation gap suggests overfitting."
    elif gap < 0.03 and train_accuracy < chance + 0.15:
        fit = "Both accuracies are low with a small gap, suggesting underfitting."
    else:
        fit = "The train-validation gap is modest; it does not by itself show severe overfitting."
    print(
        f"v{version} {task}: {VERSION_CHANGE[version]} {comparison} "
        f"Best validation accuracy={val_accuracy:.4f}; "
        f"train-validation gap={gap:.4f}. {fit}"
    )


def run_task(
    root: Path,
    task: str,
    last_version: int,
    epochs: int,
    device: torch.device,
) -> dict[int, pd.DataFrame]:
    X, y, X_test, test_ids = load_task(task, data_dir=root / "data")
    X_train, X_val, y_train, y_val = split_train_val(X, y, seed=SEED)
    X_train, X_val, _ = standardize(X_train, X_val, X_test)
    output_size = len(CLASS_NAMES[task])
    previous_accuracy: Optional[float] = None
    selected_configs: dict[int, tuple[ModelConfig, int]] = {}
    version_results: dict[int, tuple[ModelConfig, TrainingResult]] = {}
    test_predictions_by_version: dict[int, pd.DataFrame] = {}

    for version in range(1, last_version + 1):
        if version == 1:
            base = ModelConfig(
                hidden_sizes=(256, 64),
                optimizer="sgd",
                learning_rate=0.01,
                batch_size=128,
                momentum=0.9,
            )
        elif version == 2:
            base = ModelConfig(hidden_sizes=(256, 64), optimizer="adam")
        elif version == 3:
            base = replace(version_results[2][0], dropout=0.0, weight_decay=0.0, early_stopping=True)
        else:
            base = replace(
                version_results[3][0],
                hidden_sizes=(256, 256),
                early_stopping=False,
            )

        configs = one_factor_candidates(version, base)
        config, validation_run, sweep_rows, sweep_time = _choose(
            configs,
            X_train,
            y_train,
            X_val,
            y_val,
            output_size,
            epochs,
            SEED + version,
            device,
            version,
        )
        for row in sweep_rows:
            row["task"] = task
        _plot_sweeps(
            sweep_rows,
            version,
            task,
            root
            / "results"
            / "nn"
            / f"nn_v{version}_{task}_sweeps.png",
        )
        validation_probabilities = predict_probabilities(
            validation_run.model, X_val, config.batch_size, device
        )
        validation_predictions = validation_probabilities.argmax(axis=1)
        best_epoch = validation_run.best_epoch
        selected_configs[version] = (config, best_epoch)
        version_results[version] = (config, validation_run)
        training_accuracy = float(
            validation_run.history[best_epoch - 1]["train_accuracy"]
        )
        val_accuracy = validation_run.best_accuracy
        history = pd.DataFrame(validation_run.history)
        refit_started = time.perf_counter()
        X_all, _, X_test_final = standardize(X, X, X_test)
        final_run = _fit_final(
            config,
            X_all,
            y,
            output_size,
            best_epoch,
            SEED + version + 100,
            device,
        )
        refit_time = time.perf_counter() - refit_started
        parameter_total = parameter_count(final_run.model)
        prediction_started = time.perf_counter()
        test_probabilities = predict_probabilities(
            final_run.model, X_test_final, config.batch_size, device
        )
        prediction_time = time.perf_counter() - prediction_started
        test_predictions = test_probabilities.argmax(axis=1)
        training_time = sweep_time + refit_time
        config_string = config.description()
        _save_validation(
            root, version, task, y_val, validation_predictions
        )
        test_predictions_by_version[version] = pd.DataFrame(
            {"id": test_ids, "label": test_predictions.astype(int)}
        )
        gap = training_accuracy - val_accuracy
        summary_row = {
            "version": version,
            "task": task,
            "config": config_string,
            "parameters": parameter_total,
            "training_time_seconds": training_time,
            "prediction_time_seconds": prediction_time,
            "best_val_accuracy": val_accuracy,
            "final_train_accuracy": training_accuracy,
            "train_val_gap": gap,
        }
        pd.DataFrame([summary_row]).to_csv(
            root / "results" / task / f"nn_v{version}_summary.csv", index=False
        )
        _print_summary(version, task, val_accuracy, training_accuracy, previous_accuracy)
        previous_accuracy = val_accuracy
    return test_predictions_by_version


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("image", "sentiment", "both"), default="both")
    parser.add_argument(
        "--version",
        type=int,
        choices=range(1, 5),
        default=4,
        help="Run versions 1 through this version, so later versions have validation-selected predecessors.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Maximum validation-run epochs.")
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', or a PyTorch device such as 'cuda'.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but is not available")
    _prepare_directories(root)
    tasks = ("image", "sentiment") if args.task == "both" else (args.task,)
    predictions_by_task: dict[str, dict[int, pd.DataFrame]] = {}
    for task in tasks:
        predictions_by_task[task] = run_task(
            root, task, args.version, args.epochs, device
        )

    summary_frames = []
    for task in ("image", "sentiment"):
        for version in range(1, 5):
            summary_path = (
                root / "results" / task / f"nn_v{version}_summary.csv"
            )
            if summary_path.is_file():
                summary_frames.append(pd.read_csv(summary_path))
    summaries = pd.concat(summary_frames, ignore_index=True)
    summaries.to_csv(
        root / "results" / "nn" / "nn_final_summary.csv",
        index=False,
    )
    _plot_version_accuracies(root, summaries)
    if set(tasks) == {"image", "sentiment"}:
        for version in range(1, args.version + 1):
            _build_version_submission(
                root,
                version,
                {
                    task: predictions_by_task[task][version]
                    for task in ("image", "sentiment")
                },
            )
    else:
        print(
            "Combined submissions were not rebuilt; run with --task both to "
            "generate each version's combined submission."
        )


if __name__ == "__main__":
    main()
