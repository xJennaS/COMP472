"""Combine image and sentiment predictions into a validated Kaggle submission."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


_TASK_LABEL_RANGES = {"image": (0, 49), "sentiment": (0, 2)}
_EXPECTED_ROWS_PER_TASK = 5000


def _read_ids(path: Path) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    frame = pd.read_csv(path, usecols=["id"], dtype={"id": "string"})
    ids = frame["id"]
    if ids.isna().any() or ids.eq("").any():
        raise ValueError(f"{path} contains a missing or empty ID")
    if ids.duplicated().any():
        raise ValueError(f"{path} contains duplicate IDs")
    return ids


def _read_predictions(path: Path, task: str, expected_ids: pd.Series) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required prediction file not found: {path}")
    frame = pd.read_csv(path, dtype={"id": "string"})
    if list(frame.columns) != ["id", "label"]:
        raise ValueError(f"{path} must have exactly the columns 'id,label'")
    if frame["id"].isna().any() or frame["id"].eq("").any():
        raise ValueError(f"{path} contains a missing or empty ID")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate IDs")

    expected_set = set(expected_ids)
    actual_set = set(frame["id"])
    missing = expected_set - actual_set
    unexpected = actual_set - expected_set
    if missing or unexpected:
        raise ValueError(
            f"{path} IDs do not match {task}_test.csv "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )

    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.mod(1).eq(0).all():
        raise ValueError(f"{path} labels must be integers")
    label_min, label_max = _TASK_LABEL_RANGES[task]
    if not labels.between(label_min, label_max).all():
        raise ValueError(
            f"{path} labels must be in the range {label_min}-{label_max}"
        )
    frame["label"] = labels.astype(int)
    return frame


def build_submission(
    image_predictions: Optional[Path],
    sentiment_predictions: Optional[Path],
    data_dir: Path,
    output_path: Path,
    image_placeholder: Optional[int] = None,
    sentiment_placeholder: Optional[int] = None,
) -> None:
    """Validate both prediction files and write rows in sample-submission order."""
    test_ids: Dict[str, pd.Series] = {}
    predictions: Dict[str, pd.DataFrame] = {}
    for task in _TASK_LABEL_RANGES:
        test_path = data_dir / f"{task}_test.csv"
        task_ids = _read_ids(test_path)
        if len(task_ids) != _EXPECTED_ROWS_PER_TASK:
            raise ValueError(
                f"{test_path} must contain {_EXPECTED_ROWS_PER_TASK} rows "
                f"(found {len(task_ids)})"
            )
        test_ids[task] = task_ids

    for task, path, placeholder in (
        ("image", image_predictions, image_placeholder),
        ("sentiment", sentiment_predictions, sentiment_placeholder),
    ):
        if path is not None:
            predictions[task] = _read_predictions(path, task, test_ids[task])
        elif placeholder is not None:
            label_min, label_max = _TASK_LABEL_RANGES[task]
            if not label_min <= placeholder <= label_max:
                raise ValueError(
                    f"{task} placeholder must be in the range "
                    f"{label_min}-{label_max}"
                )
            predictions[task] = pd.DataFrame(
                {"id": test_ids[task], "label": placeholder}
            )
        else:
            default_path = Path(__file__).resolve().parent / f"{task}_predictions.csv"
            raise FileNotFoundError(
                f"Required prediction file not found: {default_path}. "
                f"Provide it or use --{task}-placeholder for a pipeline test."
            )

    expected_ids = set(test_ids["image"]) | set(test_ids["sentiment"])
    sample_path = data_dir / "sample_submission.csv"
    sample_ids = _read_ids(sample_path)
    if (
        len(sample_ids) != 2 * _EXPECTED_ROWS_PER_TASK
        or set(sample_ids) != expected_ids
    ):
        raise ValueError(
            f"{sample_path} must contain each image and sentiment test ID exactly once"
        )

    all_predictions = pd.concat(predictions.values(), ignore_index=True)
    submission = sample_ids.to_frame()
    submission = submission.merge(
        all_predictions, on="id", how="left", validate="one_to_one"
    )
    if (
        submission["label"].isna().any()
        or len(submission) != 2 * _EXPECTED_ROWS_PER_TASK
    ):
        raise ValueError(
            "Combined submission is missing predictions or has an invalid row count"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} rows to {output_path}")


def main() -> None:
    repository_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--image-predictions",
        type=Path,
        help="CSV with id,label predictions for image_test.csv",
    )
    image_group.add_argument(
        "--image-placeholder",
        type=int,
        choices=range(50),
        help="Use this label for every image ID (pipeline testing only)",
    )
    sentiment_group = parser.add_mutually_exclusive_group()
    sentiment_group.add_argument(
        "--sentiment-predictions",
        type=Path,
        help="CSV with id,label predictions for sentiment_test.csv",
    )
    sentiment_group.add_argument(
        "--sentiment-placeholder",
        type=int,
        choices=range(3),
        help="Use this label for every sentiment ID (pipeline testing only)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repository_dir / "data",
        help="Directory containing the test files and sample_submission.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_dir / "submission.csv",
        help="Output CSV path (default: submission.csv in the repository root)",
    )
    args = parser.parse_args()
    build_submission(
        args.image_predictions
        or (
            repository_dir / "image_predictions.csv"
            if args.image_placeholder is None
            else None
        ),
        args.sentiment_predictions
        or (
            repository_dir / "sentiment_predictions.csv"
            if args.sentiment_placeholder is None
            else None
        ),
        args.data_dir,
        args.output,
        args.image_placeholder,
        args.sentiment_placeholder,
    )


if __name__ == "__main__":
    main()
