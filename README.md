
# PQL, write polars syntax, create SQL queries for DuckDB

This package is WIP and ultimately aim to provide a polars like API for DuckDB, with full support of Duckdb functionnalities.
Compared to `narhwals`, it aims to support more functionnality from polars AND all of those from duckdb, as pql is not limited by multiple backend compatibility.
Same story for `SQLFrame`.

## Architecture

The goal is to use as much as possible of the existing Expression/Relational API and functions from duckdb. raw SQL is only used when no other choice exist.

The `sql` folder is the internal API to create duckdb Expressions.
Roles:
    - Wraps the duckdb API to provide better naming/usage ergonomics
    - Extends the existing methods
    - Handles conversion of raw inputs to duckdb expressions.

Then the "surface" level modules are wrappers on top of this API, to provide the actual user-facing API.

## Scripts

Run scripts with `uv run -m scripts`.

- the **generate** command will create a file of functions from Duckdb database in [fns](src\pql\sql\fns.py).
- The **compare** command will create the [coverage](API_COVERAGE.md) report to compare `pql` vs `polar`s and `narwhals` API's.

## References

- [DuckDB functions](https://duckdb.org/docs/stable/sql/functions/overview)
