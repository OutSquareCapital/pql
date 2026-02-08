"""Generate typed SQL function wrappers from DuckDB introspection."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyochain as pc
import typer

from ._query import run_qry
from ._schemas import TableSchema
from ._sections import FunctionInfo, build_file


def get_data(path: Path) -> None:

    import duckdb

    qry = """--sql
    SELECT *
    FROM duckdb_functions()
    """
    return duckdb.sql(qry).pl().cast(TableSchema).write_parquet(path)


def run_pipeline(path: Path) -> str:
    df = pl.scan_parquet(path).pipe(run_qry).collect()

    typer.echo(f"Found {df.height} function signatures")
    return pc.Iter(df.iter_rows()).map_star(FunctionInfo).collect().into(build_file)
