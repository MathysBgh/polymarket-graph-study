from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlns_project.config import (
    DataConfig,
    GraphConfig,
    ModelConfig,
    OutputConfig,
    PathsConfig,
    ProjectConfig,
    SplitConfig,
)
from mlns_project.pipeline import run_pipeline_with_config
from mlns_project.synthetic import generate_synthetic_dataset


def build_test_config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        name="mlns-test",
        seed=7,
        workspace_root=tmp_path,
        config_path=tmp_path / "config.toml",
        paths=PathsConfig(
            markets_path=tmp_path / "data" / "raw" / "markets.csv",
            observations_path=tmp_path / "data" / "raw" / "observations.csv",
            duckdb_path=None,
            processed_dir=tmp_path / "data" / "processed",
            outputs_dir=tmp_path / "outputs",
        ),
        data=DataConfig(
            source="files",
            markets_table="markets",
            observations_table="observations",
            duckdb_markets_query=None,
            duckdb_observations_query=None,
            assets=("BTC", "ETH"),
            contract_types=("above", "below"),
            horizon_hours=24,
            path_window_hours=24,
            path_points=6,
            min_history_points=3,
        ),
        split=SplitConfig(train_fraction=0.60, validation_fraction=0.20, test_fraction=0.20),
        graph=GraphConfig(
            alpha=0.70,
            min_edge_weight=0.20,
            strike_scale=0.15,
            use_trajectory_similarity=True,
        ),
        model=ModelConfig(inverse_regularization=1.0, max_iter=3000),
        outputs=OutputConfig(save_intermediate=True),
    )


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    generate_synthetic_dataset(
        markets_path=config.paths.markets_path,
        observations_path=config.paths.observations_path,
        seed=config.seed,
        n_days=18,
        strikes_per_type=4,
    )

    output_paths = run_pipeline_with_config(config)

    experiment_table = pd.read_csv(output_paths["experiment_table"])
    results_table = pd.read_csv(output_paths["results_table_csv"])

    assert {"train", "validation", "test"}.issubset(set(experiment_table["split"]))
    assert experiment_table.groupby("cohort_id")["split"].nunique().max() == 1

    settlement_time = pd.to_datetime(experiment_table["settlement_time"], utc=True)
    snapshot_time = pd.to_datetime(experiment_table["snapshot_time"], utc=True)
    observation_time = pd.to_datetime(experiment_table["observation_time"], utc=True)

    assert ((settlement_time - snapshot_time) == pd.Timedelta(hours=24)).all()
    assert (observation_time <= snapshot_time).all()
    assert set(results_table["model"]) == {"Crowd baseline", "Tabular model", "Graph model"}

