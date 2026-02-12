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

from . import rel_generator
from ._func_table_analysis import analyze
from .fn_generator import get_data, run_pipeline

SQL = Path("src", "pql", "sql")

DEFAULT_OUTPUT = SQL.joinpath("fns.py")

DATA_PATH = Path("scripts", "generator", "functions.parquet")

STUB_PATH = Path(".venv", "Lib", "site-packages", "_duckdb-stubs", "__init__.pyi")
OUTPUT_PATH = SQL.joinpath("_rel.py")


PathArg = Annotated[Path, typer.Option("--path", "-p")]
CheckArg = Annotated[
    bool, typer.Option("--c", help="Check output without Ruff applying fixes")
]
app = typer.Typer()

console = Console()


@app.command()
def generate_rel(
    stub_path: PathArg = STUB_PATH,
    output_path: PathArg = OUTPUT_PATH,
    *,
    check_only: CheckArg = False,
) -> None:
    """Generate typed DuckDB function wrappers from the database."""
    content = rel_generator.generate(stub_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    console.print(Text("Generated ").append(output_path.as_posix(), style="cyan"))
    _run_ruff(check_only=check_only, output=output_path)
    console.print("Done!", style="bold green")


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
    check_only: CheckArg = False,
    profile: Annotated[
        bool, typer.Option("--p", help="Enable profiling of the pipeline")
    ] = False,
) -> None:
    """Generate typed DuckDB function wrappers from the database."""
    console.print("Fetching functions from DuckDB...")
    content = run_pipeline(DATA_PATH, profile=profile)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    console.print(Text("Generated file at ").append(output.as_posix(), style="cyan"))
    _run_ruff(check_only=check_only, output=output)
    console.print("Done!", style="bold green")


@app.command()
def analyze_funcs(path: PathArg = DATA_PATH) -> None:
    """Run analysis of the functions metadata and print results in console."""
    analyze(path)


def _check_args(*, check_only: bool) -> Iterable[str]:
    if check_only:
        return ("check", "--unsafe-fixes", "--diff")

    return ("check", "--fix", "--unsafe-fixes")


def _run_ruff(*, check_only: bool, output: Path) -> None:
    typer.echo("Running Ruff checks and format...")
    uv_args = ("uv", "run", "ruff")
    run_ruff = partial(subprocess.run, check=False)

    run_ruff((*uv_args, "format", str(output)))
    run_ruff((*uv_args, *_check_args(check_only=check_only), str(output)))


if __name__ == "__main__":
    app()
