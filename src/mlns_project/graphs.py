from __future__ import annotations

from itertools import combinations
import math

import networkx as nx
import numpy as np
import pandas as pd

from .config import ProjectConfig


def _build_path_signatures(history: pd.DataFrame, config: ProjectConfig) -> dict[str, np.ndarray | None]:
    signatures: dict[str, np.ndarray | None] = {}

    for market_id, frame in history.groupby("market_id"):
        series = frame.sort_values("timestamp")["mid_price"].tail(config.data.path_points).to_numpy(dtype=float)
        if len(series) < config.data.min_history_points:
            signatures[str(market_id)] = None
            continue

        centered = series - series.mean()
        scaled = centered / (series.std() + 1e-8)
        signatures[str(market_id)] = scaled

    return signatures


def _trajectory_similarity(
    left_signature: np.ndarray | None,
    right_signature: np.ndarray | None,
) -> float:
    if left_signature is None or right_signature is None:
        return 0.5

    length = min(len(left_signature), len(right_signature))
    if length < 2:
        return 0.5

    left = left_signature[-length:]
    right = right_signature[-length:]
    correlation = np.corrcoef(left, right)[0, 1]
    if np.isnan(correlation):
        return 0.5
    return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))


def _pair_weight(
    left_row: pd.Series,
    right_row: pd.Series,
    signatures: dict[str, np.ndarray | None],
    config: ProjectConfig,
) -> float:
    strike_gap = abs(
        float(left_row["abs_strike_to_spot_distance"]) - float(right_row["abs_strike_to_spot_distance"])
    )
    strike_similarity = math.exp(-strike_gap / max(config.graph.strike_scale, 1e-6))

    if not config.graph.use_trajectory_similarity:
        return strike_similarity

    trajectory_similarity = _trajectory_similarity(
        signatures.get(str(left_row["market_id"])),
        signatures.get(str(right_row["market_id"])),
    )
    return float(
        np.clip(
            config.graph.alpha * strike_similarity
            + (1.0 - config.graph.alpha) * trajectory_similarity,
            0.0,
            1.0,
        )
    )


def _weighted_neighbor_average(graph: nx.Graph, node: str, attribute: str) -> float:
    weights: list[float] = []
    values: list[float] = []

    for neighbor, edge_attributes in graph[node].items():
        value = graph.nodes[neighbor].get(attribute)
        if pd.isna(value):
            continue
        weights.append(float(edge_attributes.get("weight", 1.0)))
        values.append(float(value))

    if not weights:
        return float("nan")

    return float(np.average(values, weights=weights))


def build_graph_features(
    experiment_table: pd.DataFrame,
    history: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, dict[str, nx.Graph]]:
    signatures = _build_path_signatures(history, config)
    feature_rows: list[dict[str, float | str]] = []
    graphs: dict[str, nx.Graph] = {}

    for cohort_id, cohort_frame in experiment_table.groupby("cohort_id"):
        graph = nx.Graph(cohort_id=cohort_id)

        for row in cohort_frame.itertuples(index=False):
            graph.add_node(
                row.market_id,
                crowd_probability=float(row.crowd_probability),
                liquidity=float(row.liquidity),
                spread=float(row.spread),
                strike=float(row.strike),
                contract_type=row.contract_type,
                label=int(row.label),
            )

        indexed_rows = {row.market_id: row._asdict() for row in cohort_frame.itertuples(index=False)}
        for left_id, right_id in combinations(indexed_rows.keys(), 2):
            left_row = pd.Series(indexed_rows[left_id])
            right_row = pd.Series(indexed_rows[right_id])
            weight = _pair_weight(left_row, right_row, signatures, config)
            if weight >= config.graph.min_edge_weight:
                graph.add_edge(left_id, right_id, weight=weight)

        if graph.number_of_edges() > 0:
            pagerank = nx.pagerank(graph, weight="weight")
            clustering = nx.clustering(graph, weight="weight")
        else:
            uniform_score = 1.0 / max(graph.number_of_nodes(), 1)
            pagerank = {node: uniform_score for node in graph.nodes}
            clustering = {node: 0.0 for node in graph.nodes}

        weighted_degree = dict(graph.degree(weight="weight"))

        for market_id in graph.nodes:
            feature_rows.append(
                {
                    "market_id": str(market_id),
                    "graph_weighted_degree": float(weighted_degree.get(market_id, 0.0)),
                    "graph_clustering": float(clustering.get(market_id, 0.0)),
                    "graph_pagerank": float(pagerank.get(market_id, 0.0)),
                    "neighbor_mean_crowd_probability": _weighted_neighbor_average(
                        graph, market_id, "crowd_probability"
                    ),
                    "neighbor_mean_liquidity": _weighted_neighbor_average(graph, market_id, "liquidity"),
                    "neighbor_mean_spread": _weighted_neighbor_average(graph, market_id, "spread"),
                }
            )

        graphs[str(cohort_id)] = graph

    return pd.DataFrame(feature_rows), graphs

