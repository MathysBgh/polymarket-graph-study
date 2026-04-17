from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import tomllib


@dataclass(slots=True)
class PathsConfig:
    markets_path: Path
    observations_path: Path
    duckdb_path: Path | None
    processed_dir: Path
    outputs_dir: Path


@dataclass(slots=True)
class DataConfig:
    source: str
    markets_table: str
    observations_table: str
    duckdb_markets_query: str | None
    duckdb_observations_query: str | None
    assets: tuple[str, ...]
    contract_types: tuple[str, ...]
    horizon_hours: int
    path_window_hours: int
    path_points: int
    min_history_points: int


@dataclass(slots=True)
class SplitConfig:
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(slots=True)
class GraphConfig:
    alpha: float
    min_edge_weight: float
    strike_scale: float
    use_trajectory_similarity: bool


@dataclass(slots=True)
class ModelConfig:
    inverse_regularization: float
    max_iter: int


@dataclass(slots=True)
class OutputConfig:
    save_intermediate: bool


@dataclass(slots=True)
class ProjectConfig:
    name: str
    seed: int
    workspace_root: Path
    config_path: Path
    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    graph: GraphConfig
    model: ModelConfig
    outputs: OutputConfig


def _infer_workspace_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return config_path.parent


def _resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _optional_query(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def load_config(config_path: str | Path) -> ProjectConfig:
    resolved_path = Path(config_path).resolve()
    workspace_root = _infer_workspace_root(resolved_path)

    with resolved_path.open("rb") as handle:
        raw = tomllib.load(handle)

    project_section = raw.get("project", {})
    paths_section = raw.get("paths", {})
    data_section = raw.get("data", {})
    split_section = raw.get("split", {})
    graph_section = raw.get("graph", {})
    model_section = raw.get("model", {})
    output_section = raw.get("outputs", {})

    split = SplitConfig(
        train_fraction=float(split_section.get("train_fraction", 0.60)),
        validation_fraction=float(split_section.get("validation_fraction", 0.20)),
        test_fraction=float(split_section.get("test_fraction", 0.20)),
    )

    total_fraction = split.train_fraction + split.validation_fraction + split.test_fraction
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")

    raw_duckdb_path = paths_section.get("duckdb_path", "")
    duckdb_path = _resolve_path(workspace_root, raw_duckdb_path) if raw_duckdb_path else None

    return ProjectConfig(
        name=str(project_section.get("name", "mlns-project")),
        seed=int(project_section.get("seed", 42)),
        workspace_root=workspace_root,
        config_path=resolved_path,
        paths=PathsConfig(
            markets_path=_resolve_path(workspace_root, str(paths_section.get("markets_path", "data/raw/markets.csv"))),
            observations_path=_resolve_path(
                workspace_root,
                str(paths_section.get("observations_path", "data/raw/observations.csv")),
            ),
            duckdb_path=duckdb_path,
            processed_dir=_resolve_path(workspace_root, str(paths_section.get("processed_dir", "data/processed"))),
            outputs_dir=_resolve_path(workspace_root, str(paths_section.get("outputs_dir", "outputs"))),
        ),
        data=DataConfig(
            source=str(data_section.get("source", "files")).lower(),
            markets_table=str(data_section.get("markets_table", "markets")),
            observations_table=str(data_section.get("observations_table", "observations")),
            duckdb_markets_query=_optional_query(data_section.get("duckdb_markets_query")),
            duckdb_observations_query=_optional_query(data_section.get("duckdb_observations_query")),
            assets=tuple(str(asset).upper() for asset in data_section.get("assets", ["BTC", "ETH"])),
            contract_types=tuple(str(name).lower() for name in data_section.get("contract_types", ["above", "below"])),
            horizon_hours=int(data_section.get("horizon_hours", 24)),
            path_window_hours=int(data_section.get("path_window_hours", 24)),
            path_points=int(data_section.get("path_points", 6)),
            min_history_points=int(data_section.get("min_history_points", 3)),
        ),
        split=split,
        graph=GraphConfig(
            alpha=float(graph_section.get("alpha", 0.70)),
            min_edge_weight=float(graph_section.get("min_edge_weight", 0.20)),
            strike_scale=float(graph_section.get("strike_scale", 0.15)),
            use_trajectory_similarity=bool(graph_section.get("use_trajectory_similarity", True)),
        ),
        model=ModelConfig(
            inverse_regularization=float(model_section.get("inverse_regularization", 1.0)),
            max_iter=int(model_section.get("max_iter", 5000)),
        ),
        outputs=OutputConfig(save_intermediate=bool(output_section.get("save_intermediate", True))),
    )

