"""Generate typed SQL function wrappers from DuckDB introspection."""

from __future__ import annotations

import subprocess
from functools import partial
from pathlib import Path
from typing import Annotated

import duckdb
import pyochain as pc
import typer

from ._query import get_df
from ._sections import FunctionInfo, build_file

DEFAULT_OUTPUT = Path("src", "pql", "sql", "fns.py")

app = typer.Typer()


@app.command()
def generate(
    output: Annotated[Path, typer.Option("--output", "-o")] = DEFAULT_OUTPUT,
) -> None:
    """Generate typed DuckDB function wrappers."""
    typer.echo("Fetching functions from DuckDB...")
    content = _run_pipeline()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    typer.echo(f"Generated {output}")
    _run_ruff(output)
    typer.echo("Done!")


@app.command()
def debug(qry: str) -> None:
    """Explore data from the table."""
    return duckdb.sql(qry).pl().show(10)


def _run_ruff(output: Path) -> None:
    typer.echo("Running Ruff checks and format...")
    uv_args = ("uv", "run", "ruff")
    run_ruff = partial(subprocess.run, check=False)
    run_ruff((*uv_args, "check", "--fix", "--unsafe-fixes", str(output)))
    run_ruff((*uv_args, "format", str(output)))


def _run_pipeline() -> str:
    df = get_df()

    typer.echo(f"Found {df.height} function signatures")
    return (
        pc.Iter(df.iter_rows())
        .map(lambda x: FunctionInfo(*x))
        .collect()
        .into(build_file)
    )


if __name__ == "__main__":
    app()
