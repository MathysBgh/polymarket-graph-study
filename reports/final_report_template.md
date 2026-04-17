# Final Project Report

## Title

Graph Structure for Probabilistic Prediction in Polymarket Crypto Contracts

## Abstract

This project studies whether graph structure between related Polymarket crypto contracts can improve probabilistic prediction beyond the crowd-implied price alone. We focus on BTC and ETH binary contracts of type `above` and `below`, define daily asset-specific settlement cohorts, and evaluate graph-aware features under a leakage-safe chronological split. Our pipeline constructs one prediction snapshot per market at `T-24h`, derives tabular market microstructure features and within-cohort graph features, and compares three model levels: the crowd baseline, a tabular logistic regression, and a graph-augmented logistic regression. Performance is evaluated with Brier score, log-loss, and expected calibration error (ECE). The central contribution of the project is a reproducible experimental framework that isolates whether relational structure contains predictive information not already absorbed by market prices.

## 1. Introduction and Motivation

Prediction markets aggregate dispersed beliefs into prices that can be interpreted as probabilities. Standard evaluation often treats each contract independently, but related contracts that settle on the same asset and date are structurally connected. This project asks whether modeling those relationships explicitly through graph features yields better probabilistic predictions than market prices alone.

Main question:

> Does graph structure between related Polymarket crypto contracts improve probabilistic prediction beyond the crowd-implied price?

## 2. Problem Definition

- Unit of prediction: one market observed at `T-24h`
- Target: binary contract resolution outcome
- Cohort definition: `asset + settlement_date`
- Main constraint: all splits must be performed at the cohort level to avoid leakage

Each market is represented by:

- a crowd-implied probability from the latest observed price at or before `T-24h`
- tabular market features such as spread, liquidity, volume, and strike-to-spot distance
- graph features computed within its settlement cohort

## 3. Related Work

Use this section to connect the project to:

- calibration and prediction market evaluation
- financial network analysis
- graph-based learning for financial forecasting
- methodological work on leakage-safe temporal evaluation

## 4. Data

Summarize the filtered dataset used in the main experiment.

Suggested insertions:

- `outputs/tables/dataset_summary.md`
- one paragraph describing market filtering choices
- one paragraph describing the `T-24h` snapshot construction

## 5. Methodology

### 5.1 Snapshot Construction

Explain:

- how markets are filtered
- how the `T-24h` prediction horizon is defined
- how the latest observation before the horizon is selected

### 5.2 Graph Construction

Explain:

- nodes as markets inside the same cohort
- weighted edges based on strike proximity and optional trajectory similarity
- graph-derived node features

Suggested figure:

- `outputs/figures/example_cohort_graph.png`

### 5.3 Models

Compare exactly:

1. Crowd baseline
2. Tabular logistic regression
3. Graph logistic regression

## 6. Evaluation

### 6.1 Experimental Protocol

- chronological cohort-level split
- no market from the same cohort appears in multiple splits
- no feature uses information later than `T-24h`

### 6.2 Metrics

- Brier score
- log-loss
- ECE

### 6.3 Main Results

Suggested insertions:

- `outputs/tables/results_table.md`
- `outputs/figures/calibration_plot.png`

## 7. Discussion

Address:

- whether graph structure improved predictive performance
- whether any gains were small but consistent
- whether the crowd price already absorbs most of the available information
- what limitations remain in the current graph design

## 8. Conclusion

State clearly:

- what the project found
- whether the graph signal was additive or not
- what the next extension would be if more time were available

## References

Add the final bibliography here in the required conference style.

