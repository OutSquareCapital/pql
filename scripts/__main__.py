"""script entry point.

Run with: `uv run -m scripts`
"""

import subprocess
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Annotated

import typer

from .generator import get_data, run_pipeline

DEFAULT_OUTPUT = Path("src", "pql", "sql", "fns.py")

DATA_PATH = Path("scripts", "generator", "functions.parquet")
PathArg = Annotated[Path, typer.Option("--path", "-p")]
app = typer.Typer()


@app.command()
def get_functions(path: PathArg = DATA_PATH) -> None:
    """Fetch function metadata from DuckDB and store as parquet at `scripts/generator/functions.parquet`."""
    get_data(path)

    typer.echo(f"Fetched function metadata and stored at {path}")


@app.command()
def compare() -> None:
    """Run the comparison between polars/narwhals and pql and generate markdown report at the repo root."""
    from .comparator import get_comparisons

    Path("API_COVERAGE.md").write_text(get_comparisons(), encoding="utf-8")


@app.command()
def generate(
    output: PathArg = DEFAULT_OUTPUT,
    *,
    ruff_fix: Annotated[bool, typer.Option("--fix/--check-only")] = True,
) -> None:
    """Generate typed DuckDB function wrappers from the database."""

    def _check_args() -> Iterable[str]:
        if ruff_fix:
            return ("check", "--fix", "--unsafe-fixes")
        return ("check", "--unsafe-fixes", "--diff")

    def _run_ruff() -> None:
        typer.echo("Running Ruff checks and format...")
        uv_args = ("uv", "run", "ruff")
        run_ruff = partial(subprocess.run, check=False)
        run_ruff((*uv_args, *_check_args(), str(output)))
        run_ruff((*uv_args, "format", str(output)))

    typer.echo("Fetching functions from DuckDB...")
    content = run_pipeline(DATA_PATH)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    typer.echo(f"Generated {output}")
    _run_ruff()
    typer.echo("Done!")


if __name__ == "__main__":
    app()
