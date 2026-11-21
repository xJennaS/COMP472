"""Create report-ready analysis from saved neural-network validation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report


def top_image_confusions(predictions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the most frequent directed true-to-predicted image-class errors."""
    mistakes = predictions.loc[
        predictions["true_label"] != predictions["predicted_label"]
    ]
    counts = (
        mistakes.groupby(["true_label", "predicted_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "true_label", "predicted_label"], ascending=[False, True, True])
        .head(top_n)
    )
    return counts


def compare_validation_errors(
    neural: pd.DataFrame, other: pd.DataFrame, model_name: str
) -> dict[str, int | str]:
    """Count NN fixes and overlapping mistakes after aligning validation indices."""
    def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.rename(
            columns={
                "validation_index": "index",
                "actual_label": "true_label",
            }
        )

    neural = standardize_columns(neural)
    other = standardize_columns(other)
    required = {"index", "true_label", "predicted_label"}
    if not required.issubset(neural.columns) or not required.issubset(other.columns):
        raise ValueError(
            f"{model_name} validation predictions must contain index,true_label,predicted_label"
        )
    merged = neural.merge(
        other,
        on="index",
        how="inner",
        suffixes=("_nn", "_other"),
        validate="one_to_one",
    )
    if len(merged) != len(neural) or len(merged) != len(other):
        raise ValueError(f"{model_name} does not cover the same validation split")
    if not (merged["true_label_nn"] == merged["true_label_other"]).all():
        raise ValueError(f"{model_name} true labels do not match neural-net labels")
    nn_wrong = merged["predicted_label_nn"] != merged["true_label_nn"]
    other_wrong = merged["predicted_label_other"] != merged["true_label_other"]
    return {
        "model": model_name,
        "nn_fixes_other_mistake": int((~nn_wrong & other_wrong).sum()),
        "other_fixes_nn_mistake": int((nn_wrong & ~other_wrong).sum()),
        "both_wrong_same_row": int((nn_wrong & other_wrong).sum()),
        "both_wrong_same_prediction": int(
            (
                nn_wrong
                & other_wrong
                & (merged["predicted_label_nn"] == merged["predicted_label_other"])
            ).sum()
        ),
    }


def analyze(root: Path) -> None:
    """Write version comparison, class diagnostics, and teammate comparisons."""
    result_dir = root / "results"
    prediction_dir = root / "predictions"
    output_dir = result_dir / "nn"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "nn_final_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Run experiments before analysis: {summary_path}")
    summary = pd.read_csv(summary_path)
    summary = summary.drop_duplicates(["version", "task"], keep="last")
    comparison = summary.pivot(
        index="version", columns="task", values="best_val_accuracy"
    ).reset_index()
    comparison.to_csv(output_dir / "nn_version_comparison.csv", index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    for task in ("image", "sentiment"):
        rows = summary.loc[summary["task"] == task].sort_values("version")
        axis.plot(rows["version"], rows["best_val_accuracy"], marker="o", label=task)
    axis.set(
        title="Neural-network validation accuracy by version",
        xlabel="Version",
        ylabel="Validation accuracy",
        xticks=[1, 2, 3, 4],
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "nn_validation_accuracy_by_version.png", dpi=150)
    plt.close(figure)

    image_path = result_dir / "image" / "nn_v4_validation.csv"
    if image_path.is_file():
        image_predictions = pd.read_csv(image_path)
        image_predictions = image_predictions.rename(
            columns={"actual_label": "true_label"}
        )
        top_image_confusions(image_predictions).to_csv(
            output_dir / "nn_image_top_confused_pairs.csv", index=False
        )

    sentiment_path = result_dir / "sentiment" / "nn_v4_validation.csv"
    if sentiment_path.is_file():
        sentiment_predictions = pd.read_csv(sentiment_path).rename(
            columns={"actual_label": "true_label"}
        )
        report = classification_report(
            sentiment_predictions["true_label"],
            sentiment_predictions["predicted_label"],
            labels=[0, 1, 2],
            target_names=["Negative", "Neutral", "Positive"],
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().loc[
            ["Negative", "Neutral", "Positive"]
        ].to_csv(output_dir / "nn_sentiment_per_class_f1.csv", index_label="class")
        f1_values = {
            name: report[name]["f1-score"] for name in ("Negative", "Neutral", "Positive")
        }
        # Lowest F1 is the hardest class; report Neutral explicitly for the project question.
        hardest = min(f1_values, key=f1_values.get)
        print(
            "Sentiment class F1: "
            + ", ".join(f"{name}={score:.4f}" for name, score in f1_values.items())
            + f". Hardest class by F1: {hardest}; "
            + ("Neutral is hardest." if hardest == "Neutral" else "Neutral is not hardest.")
        )

    for task in ("image", "sentiment"):
        nn_path = result_dir / task / "nn_v4_validation.csv"
        if not nn_path.is_file():
            continue
        neural = pd.read_csv(nn_path)
        comparisons = []
        for model_path in prediction_dir.glob(f"*_{task}_val.csv"):
            if model_path == nn_path or model_path.name.startswith("nn_"):
                continue
            other = pd.read_csv(model_path)
            comparisons.append(
                compare_validation_errors(neural, other, model_path.stem)
            )
        if comparisons:
            pd.DataFrame(comparisons).to_csv(
                output_dir / f"nn_v4_{task}_teammate_error_comparison.csv",
                index=False,
            )
            print(f"{task}: wrote comparisons against {len(comparisons)} teammate model(s).")
        else:
            print(f"{task}: no teammate validation prediction files found.")
    print(f"Report tables written under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing results/ and optional teammate predictions/.",
    )
    args = parser.parse_args()
    analyze(args.root)


if __name__ == "__main__":
    main()
