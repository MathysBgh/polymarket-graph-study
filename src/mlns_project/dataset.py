from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig


def filter_markets(markets: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    filtered = markets.loc[
        markets["asset"].isin(config.data.assets)
        & markets["contract_type"].isin(config.data.contract_types)
    ].copy()

    filtered["settlement_date"] = filtered["settlement_time"].dt.floor("D")
    filtered["cohort_id"] = (
        filtered["asset"] + "_" + filtered["settlement_date"].dt.strftime("%Y-%m-%d")
    )
    filtered["snapshot_time"] = filtered["settlement_time"] - pd.to_timedelta(
        config.data.horizon_hours, unit="h"
    )
    return filtered


def build_snapshot_dataset(
    markets: pd.DataFrame,
    observations: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_markets = filter_markets(markets, config)
    if filtered_markets.empty:
        raise ValueError("No markets remain after filtering by asset and contract type.")

    joined = observations.merge(
        filtered_markets[
            ["market_id", "cohort_id", "snapshot_time", "settlement_time", "strike", "asset", "contract_type", "label"]
        ],
        on="market_id",
        how="inner",
    )

    eligible_for_snapshot = joined.loc[joined["timestamp"] <= joined["snapshot_time"]].copy()
    if eligible_for_snapshot.empty:
        raise ValueError("No observations are available at or before the prediction horizon.")

    snapshots = (
        eligible_for_snapshot.sort_values(["market_id", "timestamp"])
        .groupby("market_id", as_index=False)
        .tail(1)
        .rename(columns={"timestamp": "observation_time"})
    )

    experiment_base = filtered_markets.merge(
        snapshots[
            [
                "market_id",
                "observation_time",
                "mid_price",
                "spread",
                "liquidity",
                "volume",
                "reference_spot_price",
            ]
        ],
        on="market_id",
        how="inner",
    )

    experiment_base["crowd_probability"] = experiment_base["mid_price"].clip(1e-4, 1 - 1e-4)
    experiment_base["strike_to_spot_distance"] = (
        experiment_base["strike"] - experiment_base["reference_spot_price"]
    ) / experiment_base["reference_spot_price"]
    experiment_base["abs_strike_to_spot_distance"] = experiment_base["strike_to_spot_distance"].abs()

    history_window = pd.to_timedelta(config.data.path_window_hours, unit="h")
    history = eligible_for_snapshot.loc[
        eligible_for_snapshot["timestamp"] >= eligible_for_snapshot["snapshot_time"] - history_window
    ].copy()

    return experiment_base, history


def _compute_split_sizes(total: int, config: ProjectConfig) -> list[int]:
    raw_sizes = np.array(
        [
            config.split.train_fraction,
            config.split.validation_fraction,
            config.split.test_fraction,
        ]
    ) * total
    sizes = np.floor(raw_sizes).astype(int)
    remainder = total - int(sizes.sum())

    order = np.argsort(raw_sizes - sizes)[::-1]
    for index in order[:remainder]:
        sizes[index] += 1

    if total >= 3:
        for required_index in range(3):
            if sizes[required_index] == 0:
                donor = int(np.argmax(sizes))
                if sizes[donor] <= 1:
                    continue
                sizes[donor] -= 1
                sizes[required_index] += 1

    return sizes.tolist()


def assign_cohort_splits(experiment_table: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    cohorts = (
        experiment_table.groupby("cohort_id", as_index=False)["settlement_time"]
        .min()
        .sort_values("settlement_time")
        .reset_index(drop=True)
    )
    split_sizes = _compute_split_sizes(len(cohorts), config)

    split_labels = (
        ["train"] * split_sizes[0]
        + ["validation"] * split_sizes[1]
        + ["test"] * split_sizes[2]
    )
    cohorts["split"] = split_labels[: len(cohorts)]

    return experiment_table.merge(cohorts[["cohort_id", "split"]], on="cohort_id", how="left")


def validate_no_cohort_leakage(experiment_table: pd.DataFrame) -> None:
    leakage = experiment_table.groupby("cohort_id")["split"].nunique()
    if (leakage > 1).any():
        raise ValueError("Cohort leakage detected across train/validation/test splits.")


def validate_snapshot_horizon(experiment_table: pd.DataFrame, horizon_hours: int) -> None:
    expected_snapshot = experiment_table["settlement_time"] - pd.to_timedelta(horizon_hours, unit="h")
    if not experiment_table["snapshot_time"].equals(expected_snapshot):
        raise ValueError("Snapshot times do not match the configured horizon.")

    if (experiment_table["observation_time"] > experiment_table["snapshot_time"]).any():
        raise ValueError("Some snapshot observations occur after the prediction horizon.")

