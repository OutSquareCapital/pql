
# PQL, write polars syntax, create SQL queries for DuckDB

This package is WIP and ultimately aim to provide a polars like API for DuckDB, with full support of Duckdb functionnalities.
Compared to `narhwals`, it aims to support more functionnality from polars AND all of those from duckdb, as pql is not limited by multiple backend compatibility.
Same story for `SQLFrame`.

## Scripts

Run scripts with `uv run -m scripts`.

- the **generate** command will create a file of functions from Duckdb database in [fns](src\pql\sql\fns.py).
- The **compare** command will create the [coverage](API_COVERAGE.md) report to compare `pql` vs `polar`s and `narwhals` API's.
