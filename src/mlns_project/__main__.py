"""Package entrypoint."""

from pathlib import Path

from .pipeline import run_pipeline


def main() -> None:
    run_pipeline(Path("config/project.synthetic.toml"))


if __name__ == "__main__":
    main()

