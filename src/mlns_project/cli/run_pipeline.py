from __future__ import annotations

import argparse
from pathlib import Path

from ..pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MLNS project pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/project.synthetic.toml"),
        help="Path to the project TOML configuration file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_paths = run_pipeline(args.config)
    print("Pipeline completed successfully.")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

