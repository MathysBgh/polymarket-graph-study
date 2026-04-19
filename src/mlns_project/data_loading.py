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


_SQLITE_MARKETS_QUERY = """
SELECT
    condition_id      AS market_id,
    UPPER(asset)      AS asset,
    LOWER(direction)  AS contract_type,
    threshold         AS strike,
    settlement_date   AS settlement_time,
    outcome           AS label
FROM markets
WHERE asset IN ({assets})
  AND direction IN ({directions})
  AND outcome IS NOT NULL
"""

_SQLITE_OBSERVATIONS_QUERY = """
WITH price_hours AS (
    SELECT
        mp.condition_id,
        m.asset,
        CAST(mp.timestamp AS INTEGER)           AS ts_seconds,
        (CAST(mp.timestamp AS INTEGER) / 3600) * 3600 AS hour_seconds,
        mp.yes_price,
        mp.no_price,
        mp.volume,
        mp.trade_count
    FROM market_prices AS mp
    JOIN markets AS m USING (condition_id)
    WHERE m.asset IN ({assets})
      AND m.direction IN ({directions})
      AND mp.yes_price IS NOT NULL
)
SELECT
    ph.condition_id                                                            AS market_id,
    ph.ts_seconds                                                              AS timestamp_seconds,
    ph.yes_price                                                               AS mid_price,
    CASE
        WHEN ph.no_price IS NULL THEN 0.0
        ELSE ABS(1.0 - ph.yes_price - ph.no_price)
    END                                                                        AS spread,
    COALESCE(ph.volume, 0.0)                                                   AS liquidity,
    COALESCE(ph.trade_count, 0)                                                AS volume,
    o.close                                                                    AS reference_spot_price
FROM price_hours AS ph
LEFT JOIN ohlcv AS o
       ON o.asset = ph.asset
      AND o.timestamp = ph.hour_seconds * 1000
WHERE o.close IS NOT NULL
"""


def _format_sql_list(values: tuple[str, ...]) -> str:
    if not values:
        raise ValueError("SQLite source needs at least one asset and one contract type.")
    return ", ".join(f"'{value}'" for value in values)


def _load_from_sqlite(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.paths.duckdb_path is None:
        raise ValueError("SQLite source was selected but no duckdb_path was provided.")

    import sqlite3

    assets_sql = _format_sql_list(config.data.assets)
    directions_sql = _format_sql_list(config.data.contract_types)
    markets_query = _SQLITE_MARKETS_QUERY.format(assets=assets_sql, directions=directions_sql)
    observations_query = _SQLITE_OBSERVATIONS_QUERY.format(
        assets=assets_sql, directions=directions_sql
    )

    with sqlite3.connect(f"file:{config.paths.duckdb_path}?mode=ro", uri=True) as connection:
        markets = pd.read_sql_query(markets_query, connection)
        observations = pd.read_sql_query(observations_query, connection)

    observations["timestamp"] = pd.to_datetime(
        observations.pop("timestamp_seconds"), unit="s", utc=True
    )
    return _normalize_markets(markets), _normalize_observations(observations)


def load_canonical_data(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.data.source == "files":
        return _load_from_files(config)
    if config.data.source == "duckdb":
        return _load_from_duckdb(config)
    if config.data.source == "sqlite":
        return _load_from_sqlite(config)
    raise ValueError(f"Unsupported data source: {config.data.source}")

