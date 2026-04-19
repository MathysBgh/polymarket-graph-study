# Polymarket Graph Study (MLNS 2026)

Code and final report for the CentraleSupélec MLNS 2026 final project:

> **Can graph structure between related Polymarket crypto contracts improve probabilistic prediction beyond the crowd-implied price?**

Team: Alexandre Dalban, Mathys Bagnah, Wacil Lakbir.

## Layout

```text
.
|-- config/project.toml          # pipeline configuration
|-- data/
|   |-- raw/                     # input databases (gitignored, see Releases)
|   `-- processed/               # generated tables (gitignored)
|-- notebooks/                   # walkthrough notebook with rendered outputs
|-- outputs/
|   |-- figures/                 # cohort graph, calibration plot
|   `-- tables/                  # dataset summary, results table
|-- reports/final_report.tex     # ICLR 2025 final report source
|-- src/mlns_project/            # pipeline package
|   `-- cli/run_pipeline.py      # main entrypoint
|-- MLNS_Project_Proposal.pdf    # project proposal (deliverable)
`-- pyproject.toml
```

## Reproducing the experiment

1. Create a Python 3.11+ environment and install the package:

   ```bash
   py -3 -m pip install -e .
   ```

2. Download the dataset from this repository's GitHub Releases page and unpack
   into `data/raw/`:

   ```bash
   gh release download v1.0-data --pattern "backtest_sample.db.gz"
   gunzip backtest_sample.db.gz
   mv backtest_sample.db data/raw/
   ```

   The release also ships the optional Deribit options/futures parquet files
   used for supplementary analysis (`options_5m.parquet`, `futures_5m.parquet`,
   `ohlcv_5m.parquet`, `funding_8h.parquet`, `dvol_5m.parquet`).

3. Run the pipeline:

   ```bash
   py -3 -m mlns_project.cli.run_pipeline --config config/project.toml
   ```

   This regenerates `data/processed/{experiment_table,predictions,metrics_by_split}.csv`
   and the figures and tables under `outputs/`.

4. Open `notebooks/01_experiment_walkthrough.ipynb` to inspect the results
   interactively.

## Final report

Source in `reports/final_report.tex` (ICLR 2025 template, drop into Overleaf
or compile locally with `latexmk -pdf reports/final_report.tex`).

## Data schema

The pipeline expects the `taut-arb-backtest` v2.0 SQLite schema (https://github.com/ADnocap/taut-arb-backtest). The relevant
tables are `markets` (one row per resolved Polymarket binary contract),
`market_prices` (YES/NO time series), and `ohlcv` (hourly Deribit perpetual
candles for the spot anchor). The SQL adapter in
`src/mlns_project/data_loading.py` maps these to the canonical columns
(`market_id`, `asset`, `contract_type`, `strike`, `settlement_time`, `label`
for markets; `mid_price`, `spread`, `liquidity`, `volume`,
`reference_spot_price` for observations).
