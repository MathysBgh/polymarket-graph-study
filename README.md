# MLNS Project Workspace

This repository is a VS Code-first Python workspace for the MLNS final project:

**Does graph structure between related Polymarket crypto contracts improve probabilistic prediction beyond the crowd-implied price?**

The workspace is designed to stay usable even when the real data is not stored locally yet. It includes:

- a clean `src/` package for the full pipeline
- a central TOML configuration entrypoint
- a synthetic data generator for end-to-end verification
- notebook and report assets written in English
- output generation for the final experiment table, descriptive tables, graph figures, calibration plots, and ablation results

## Project layout

```text
.
|-- config/
|   |-- project.example.toml
|   `-- project.synthetic.toml
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|-- outputs/
|   |-- figures/
|   `-- tables/
|-- reports/
|-- src/mlns_project/
|   |-- cli/
|   `-- ...
`-- tests/
```

## Expected raw data schema

The file-based pipeline expects two canonical tables:

### `markets`

- `market_id`
- `asset`
- `contract_type`
- `strike`
- `settlement_time`
- `label`

### `observations`

- `market_id`
- `timestamp`
- `mid_price`
- `spread`
- `liquidity`
- `volume`
- `reference_spot_price`

You can provide these as `.csv` or `.parquet` files, or load them from DuckDB with SQL aliases that produce the same canonical columns.

## Quick start in VS Code

1. Create and activate a Python environment.
2. Install the package:

```powershell
py -3 -m pip install -e .[dev]
```

3. Generate a synthetic dataset for local verification:

```powershell
py -3 -m mlns_project.cli.generate_synthetic --config config/project.synthetic.toml
```

4. Run the full pipeline:

```powershell
py -3 -m mlns_project.cli.run_pipeline --config config/project.synthetic.toml
```

5. Open `notebooks/01_experiment_walkthrough.ipynb` in VS Code to inspect outputs.

## Using the real dataset later

- Keep the code unchanged.
- Copy `config/project.example.toml` to a working config file.
- Point `markets_path` / `observations_path`, or `duckdb_path`, to the real data source.
- If the DuckDB schema differs from the canonical schema, update the SQL queries in the config so they return the required aliases.

## Main outputs

- `data/processed/experiment_table.csv`
- `outputs/tables/dataset_summary.csv`
- `outputs/tables/results_table.csv`
- `outputs/figures/example_cohort_graph.png`
- `outputs/figures/calibration_plot.png`

