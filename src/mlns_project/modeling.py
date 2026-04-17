from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ProjectConfig
from .evaluation import evaluate_predictions

BASE_NUMERIC_FEATURES = [
    "crowd_probability",
    "spread",
    "liquidity",
    "volume",
    "strike_to_spot_distance",
]

GRAPH_NUMERIC_FEATURES = [
    "graph_weighted_degree",
    "graph_clustering",
    "graph_pagerank",
    "neighbor_mean_crowd_probability",
    "neighbor_mean_liquidity",
    "neighbor_mean_spread",
]

CATEGORICAL_FEATURES = ["asset", "contract_type"]

PREDICTION_COLUMNS = {
    "Crowd baseline": "crowd_baseline",
    "Tabular model": "tabular_model",
    "Graph model": "graph_model",
}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


FEATURE_SPECS = {
    "tabular_model": FeatureSpec(
        numeric=tuple(BASE_NUMERIC_FEATURES),
        categorical=tuple(CATEGORICAL_FEATURES),
    ),
    "graph_model": FeatureSpec(
        numeric=tuple(BASE_NUMERIC_FEATURES + GRAPH_NUMERIC_FEATURES),
        categorical=tuple(CATEGORICAL_FEATURES),
    ),
}


def _build_logistic_pipeline(feature_spec: FeatureSpec, config: ProjectConfig) -> Pipeline:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(feature_spec.numeric)),
            ("categorical", categorical_transformer, list(feature_spec.categorical)),
        ]
    )

    classifier = LogisticRegression(
        C=max(config.model.inverse_regularization, 1e-6),
        max_iter=config.model.max_iter,
        random_state=config.seed,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def run_models(
    experiment_table: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = experiment_table.copy()
    predictions["crowd_baseline"] = predictions["crowd_probability"].clip(1e-6, 1 - 1e-6)

    train_frame = predictions.loc[predictions["split"] == "train"].copy()
    if train_frame.empty:
        raise ValueError("The train split is empty. Adjust the split configuration or dataset size.")

    for model_name, feature_spec in FEATURE_SPECS.items():
        pipeline = _build_logistic_pipeline(feature_spec, config)
        feature_columns = list(feature_spec.numeric + feature_spec.categorical)
        pipeline.fit(train_frame[feature_columns], train_frame["label"].astype(int))
        predictions[model_name] = pipeline.predict_proba(predictions[feature_columns])[:, 1]

    metrics = evaluate_predictions(predictions, PREDICTION_COLUMNS)
    return predictions, metrics

