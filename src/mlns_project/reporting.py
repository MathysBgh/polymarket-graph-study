from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .config import ProjectConfig
from .evaluation import build_calibration_frame
from .modeling import PREDICTION_COLUMNS


def _format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    headers = list(dataframe.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in dataframe.itertuples(index=False):
        lines.append("| " + " | ".join(_format_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def _write_markdown_table(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.write_text(_dataframe_to_markdown(dataframe), encoding="utf-8")


def write_dataset_summary(experiment_table: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    settlement_min = experiment_table["settlement_time"].min()
    settlement_max = experiment_table["settlement_time"].max()
    markets_per_cohort = experiment_table.groupby("cohort_id").size()

    summary = pd.DataFrame(
        [
            {"metric": "markets", "value": int(len(experiment_table))},
            {"metric": "cohorts", "value": int(experiment_table["cohort_id"].nunique())},
            {"metric": "assets", "value": ", ".join(sorted(experiment_table["asset"].unique()))},
            {
                "metric": "contract_types",
                "value": ", ".join(sorted(experiment_table["contract_type"].unique())),
            },
            {"metric": "start_settlement", "value": settlement_min.isoformat()},
            {"metric": "end_settlement", "value": settlement_max.isoformat()},
            {"metric": "avg_markets_per_cohort", "value": float(markets_per_cohort.mean())},
        ]
    )

    csv_path = output_dir / "dataset_summary.csv"
    markdown_path = output_dir / "dataset_summary.md"
    summary.to_csv(csv_path, index=False)
    _write_markdown_table(summary, markdown_path)
    return {"dataset_summary_csv": csv_path, "dataset_summary_md": markdown_path}


def write_results_table(metrics: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    test_results = metrics.loc[metrics["split"] == "test"].copy()
    test_results = test_results.sort_values("brier_score").reset_index(drop=True)

    csv_path = output_dir / "results_table.csv"
    markdown_path = output_dir / "results_table.md"
    test_results.to_csv(csv_path, index=False)
    _write_markdown_table(test_results, markdown_path)
    return {"results_table_csv": csv_path, "results_table_md": markdown_path}


def save_example_graph(
    experiment_table: pd.DataFrame,
    graphs: dict[str, nx.Graph],
    output_path: Path,
) -> Path:
    candidate_cohorts = (
        experiment_table.loc[experiment_table["split"] == "test", "cohort_id"]
        .value_counts()
        .index.tolist()
    )
    if not candidate_cohorts:
        candidate_cohorts = experiment_table["cohort_id"].value_counts().index.tolist()

    chosen_graph = None
    for cohort_id in candidate_cohorts:
        graph = graphs.get(str(cohort_id))
        if graph is not None and graph.number_of_nodes() > 0:
            chosen_graph = graph
            break

    if chosen_graph is None:
        raise ValueError("Could not find a cohort graph to visualize.")

    plt.figure(figsize=(10, 7))
    layout = nx.spring_layout(chosen_graph, seed=42, weight="weight")
    node_colors = [chosen_graph.nodes[node]["crowd_probability"] for node in chosen_graph.nodes]
    edge_widths = [2.0 * chosen_graph.edges[edge].get("weight", 1.0) for edge in chosen_graph.edges]
    labels = {
        node: f"{chosen_graph.nodes[node]['contract_type']}\n{chosen_graph.nodes[node]['strike']:.0f}"
        for node in chosen_graph.nodes
    }

    nx.draw_networkx_nodes(
        chosen_graph,
        pos=layout,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=900,
    )
    if chosen_graph.number_of_edges() > 0:
        nx.draw_networkx_edges(chosen_graph, pos=layout, width=edge_widths, alpha=0.6)
    nx.draw_networkx_labels(chosen_graph, pos=layout, labels=labels, font_size=8)
    plt.title(f"Example Cohort Graph: {chosen_graph.graph.get('cohort_id', 'unknown cohort')}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_calibration_plot(predictions: pd.DataFrame, output_path: Path) -> Path:
    plt.figure(figsize=(8, 6))

    for model_name, column_name in PREDICTION_COLUMNS.items():
        calibration_frame = build_calibration_frame(predictions, column_name, split="test")
        if calibration_frame.empty:
            continue
        plt.plot(
            calibration_frame["mean_predicted"],
            calibration_frame["empirical_rate"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.5, label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Calibration on the Test Split")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def write_outputs(
    config: ProjectConfig,
    experiment_table: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    graphs: dict[str, nx.Graph],
) -> dict[str, Path]:
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = config.paths.outputs_dir / "figures"
    table_dir = config.paths.outputs_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    experiment_path = config.paths.processed_dir / "experiment_table.csv"
    predictions_path = config.paths.processed_dir / "predictions.csv"
    metrics_path = config.paths.processed_dir / "metrics_by_split.csv"
    experiment_table.to_csv(experiment_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    output_paths: dict[str, Path] = {
        "experiment_table": experiment_path,
        "predictions": predictions_path,
        "metrics_by_split": metrics_path,
    }
    output_paths.update(write_dataset_summary(experiment_table, table_dir))
    output_paths.update(write_results_table(metrics, table_dir))
    output_paths["example_graph"] = save_example_graph(
        experiment_table,
        graphs,
        figure_dir / "example_cohort_graph.png",
    )
    output_paths["calibration_plot"] = save_calibration_plot(
        predictions,
        figure_dir / "calibration_plot.png",
    )

    manifest_path = config.paths.outputs_dir / "output_manifest.json"
    manifest = {name: str(path) for name, path in output_paths.items()}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output_paths["manifest"] = manifest_path
    return output_paths

