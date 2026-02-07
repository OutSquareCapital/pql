"""Generate typed SQL function wrappers from DuckDB introspection."""

from __future__ import annotations

import pyochain as pc
import typer

from ._query import get_df
from ._schemas import DATA_PATH, TableSchema
from ._sections import FunctionInfo, build_file


def get_data() -> None:

    import duckdb

    qry = """--sql
    SELECT *
    FROM duckdb_functions()
    """
    return duckdb.sql(qry).pl().cast(TableSchema).write_parquet(DATA_PATH)


def run_pipeline() -> str:
    df = get_df().collect()

    typer.echo(f"Found {df.height} function signatures")
    return pc.Iter(df.iter_rows()).map_star(FunctionInfo).collect().into(build_file)
