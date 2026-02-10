"""script entry point.

Run with: `uv run -m scripts`
"""

import subprocess
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text

from .generator import get_data, run_pipeline

DEFAULT_OUTPUT = Path("src", "pql", "sql", "fns.py")

DATA_PATH = Path("scripts", "generator", "functions.parquet")
PathArg = Annotated[Path, typer.Option("--path", "-p")]
app = typer.Typer()

console = Console()


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
    check_only: Annotated[bool, typer.Option("--c")] = False,
    profile: Annotated[bool, typer.Option("--p")] = False,
) -> None:
    """Generate typed DuckDB function wrappers from the database."""

    def _check_args() -> Iterable[str]:
        if check_only:
            return ("check", "--unsafe-fixes", "--diff")

        return ("check", "--fix", "--unsafe-fixes")

    def _run_ruff() -> None:
        typer.echo("Running Ruff checks and format...")
        uv_args = ("uv", "run", "ruff")
        run_ruff = partial(subprocess.run, check=False)

        run_ruff((*uv_args, "format", str(output)))
        run_ruff((*uv_args, *_check_args(), str(output)))

    console.print("Fetching functions from DuckDB...")
    content = run_pipeline(DATA_PATH, profile=profile)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    console.print(Text("Generated file at ").append(output.as_posix(), style="cyan"))
    _run_ruff()
    console.print("Done!", style="bold green")


if __name__ == "__main__":
    app()
