from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ProjectConfig

REQUIRED_MARKET_COLUMNS = {
    "market_id",
    "asset",
    "contract_type",
    "strike",
    "settlement_time",
    "label",
}

REQUIRED_OBSERVATION_COLUMNS = {
    "market_id",
    "timestamp",
    "mid_price",
    "spread",
    "liquidity",
    "volume",
    "reference_spot_price",
}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format for {path.name}. Use CSV or Parquet.")


def _coerce_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0})
    if mapped.isna().any():
        raise ValueError("Could not coerce all labels to binary values.")
    return mapped.astype(int)


def _normalize_markets(markets: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_MARKET_COLUMNS.difference(markets.columns)
    if missing:
        raise ValueError(f"Markets table is missing columns: {sorted(missing)}")

    normalized = markets.copy()
    normalized["market_id"] = normalized["market_id"].astype(str)
    normalized["asset"] = normalized["asset"].astype(str).str.upper()
    normalized["contract_type"] = normalized["contract_type"].astype(str).str.lower()
    normalized["strike"] = pd.to_numeric(normalized["strike"], errors="raise")
    normalized["settlement_time"] = pd.to_datetime(normalized["settlement_time"], utc=True)
    normalized["label"] = _coerce_labels(normalized["label"])
    return normalized


def _normalize_observations(observations: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_OBSERVATION_COLUMNS.difference(observations.columns)
    if missing:
        raise ValueError(f"Observations table is missing columns: {sorted(missing)}")

    normalized = observations.copy()
    normalized["market_id"] = normalized["market_id"].astype(str)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)

    for column in ["mid_price", "spread", "liquidity", "volume", "reference_spot_price"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(
        subset=["timestamp", "mid_price", "spread", "liquidity", "volume", "reference_spot_price"]
    )
    return normalized


def _default_markets_query(table_name: str) -> str:
    return (
        f"SELECT market_id, asset, contract_type, strike, settlement_time, label "
        f"FROM {table_name}"
    )


def _default_observations_query(table_name: str) -> str:
    return (
        "SELECT market_id, timestamp, mid_price, spread, liquidity, volume, reference_spot_price "
        f"FROM {table_name}"
    )


def _load_from_duckdb(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.paths.duckdb_path is None:
        raise ValueError("DuckDB source was selected but no duckdb_path was provided.")

    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("duckdb is required to load data from a DuckDB database.") from exc

    markets_query = config.data.duckdb_markets_query or _default_markets_query(config.data.markets_table)
    observations_query = config.data.duckdb_observations_query or _default_observations_query(
        config.data.observations_table
    )

    with duckdb.connect(str(config.paths.duckdb_path), read_only=True) as connection:
        markets = connection.execute(markets_query).fetch_df()
        observations = connection.execute(observations_query).fetch_df()

    return _normalize_markets(markets), _normalize_observations(observations)


def _load_from_files(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    markets = _read_table(config.paths.markets_path)
    observations = _read_table(config.paths.observations_path)
    return _normalize_markets(markets), _normalize_observations(observations)


def load_canonical_data(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.data.source == "files":
        return _load_from_files(config)
    if config.data.source == "duckdb":
        return _load_from_duckdb(config)
    raise ValueError(f"Unsupported data source: {config.data.source}")

