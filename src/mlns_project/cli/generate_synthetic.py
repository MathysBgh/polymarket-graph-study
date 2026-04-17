from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..synthetic import generate_synthetic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic Polymarket-style dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/project.synthetic.toml"),
        help="Path to the project TOML configuration file.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of synthetic settlement days to generate.",
    )
    parser.add_argument(
        "--strikes-per-type",
        type=int,
        default=6,
        help="Number of strike levels per contract type and asset.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    markets, observations = generate_synthetic_dataset(
        markets_path=config.paths.markets_path,
        observations_path=config.paths.observations_path,
        seed=config.seed,
        n_days=args.days,
        strikes_per_type=args.strikes_per_type,
    )
    print("Synthetic dataset generated successfully.")
    print(f"markets rows: {len(markets)}")
    print(f"observations rows: {len(observations)}")
    print(f"markets path: {config.paths.markets_path}")
    print(f"observations path: {config.paths.observations_path}")


if __name__ == "__main__":
    main()

