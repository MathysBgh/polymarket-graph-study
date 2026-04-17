from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig, load_config
from .data_loading import load_canonical_data
from .dataset import (
    assign_cohort_splits,
    build_snapshot_dataset,
    validate_no_cohort_leakage,
    validate_snapshot_horizon,
)
from .graphs import build_graph_features
from .modeling import run_models
from .reporting import write_outputs


def run_pipeline(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    return run_pipeline_with_config(config)


def run_pipeline_with_config(config: ProjectConfig) -> dict[str, Path]:
    markets, observations = load_canonical_data(config)
    experiment_table, history = build_snapshot_dataset(markets, observations, config)
    experiment_table = assign_cohort_splits(experiment_table, config)

    graph_features, graphs = build_graph_features(experiment_table, history, config)
    experiment_table = experiment_table.merge(graph_features, on="market_id", how="left")

    validate_no_cohort_leakage(experiment_table)
    validate_snapshot_horizon(experiment_table, config.data.horizon_hours)

    predictions, metrics = run_models(experiment_table, config)
    return write_outputs(config, experiment_table, predictions, metrics, graphs)

