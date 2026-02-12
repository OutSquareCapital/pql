
# PQL, write polars syntax, create SQL queries for DuckDB

This package is WIP and ultimately aim to provide a polars like API for DuckDB, with full support of Duckdb functionnalities.
Compared to `narhwals`, it aims to support more functionnality from polars AND all of those from duckdb, as pql is not limited by multiple backend compatibility.
Same story for `SQLFrame`.

## Architecture

Currently, the package is structured in two layers:

- the **SQL layer** in `src/pql/sql` is the core of the package, where the SQL query is built and generated. It wraps the duckdb python API and provides a more user friendly interface to build SQL queries, as well as expanding significantly the API with more functions and features, thanks to code generation scripts from duckdb catalog for example.
- the **polars layer** in `src/pql/{_frame.py, _expr.py}` is a wrapper around the SQL layer, that provides a polars like API to build SQL queries.
It is designed to be as close as possible to the polars API, while still providing access to all the features of the SQL layer.
This is the public API of the package, and the one that users should use to build their queries.

## Scripts

Run scripts with `uv run -m scripts`.

- the **generate** command will create a file of functions from Duckdb database in [fns](src\pql\sql\fns.py).
- The **compare** command will create the [coverage](API_COVERAGE.md) report to compare `pql` vs `polar`s and `narwhals` API's.

## References

- [DuckDB functions](https://duckdb.org/docs/stable/sql/functions/overview)
