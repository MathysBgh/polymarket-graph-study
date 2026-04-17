from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        weight = float(mask.mean())
        ece += abs(confidence - accuracy) * weight

    return float(ece)


def evaluate_predictions(
    predictions: pd.DataFrame,
    prediction_columns: dict[str, str],
) -> pd.DataFrame:
    records: list[dict[str, float | str | int]] = []

    for split in ["train", "validation", "test"]:
        split_frame = predictions.loc[predictions["split"] == split]
        if split_frame.empty:
            continue

        y_true = split_frame["label"].astype(int).to_numpy()
        for model_name, column_name in prediction_columns.items():
            y_prob = np.clip(split_frame[column_name].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
            records.append(
                {
                    "split": split,
                    "model": model_name,
                    "n_samples": int(len(split_frame)),
                    "brier_score": float(brier_score_loss(y_true, y_prob)),
                    "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
                    "ece": expected_calibration_error(y_true, y_prob),
                }
            )

    return pd.DataFrame(records)


def build_calibration_frame(
    predictions: pd.DataFrame,
    prediction_column: str,
    split: str = "test",
    n_bins: int = 10,
) -> pd.DataFrame:
    split_frame = predictions.loc[predictions["split"] == split, ["label", prediction_column]].copy()
    if split_frame.empty:
        return pd.DataFrame(columns=["bin", "count", "mean_predicted", "empirical_rate"])

    y_true = split_frame["label"].astype(int).to_numpy()
    y_prob = np.clip(split_frame[prediction_column].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1], right=True)

    records: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        records.append(
            {
                "bin": bin_index,
                "count": int(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()),
                "empirical_rate": float(y_true[mask].mean()),
            }
        )

    return pd.DataFrame(records)

