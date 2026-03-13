
# PQL, write polars syntax, create SQL queries for DuckDB

This package is **WIP** and ultimately aim to provide a polars like API for [DuckDB Python API](https://duckdb.org/docs/stable/clients/python/overview), as well as full support of the [DuckDB functions](https://duckdb.org/docs/stable/sql/functions/overview).

## Comparison to other tools

### Narwhals

[narwhals](https://github.com/narwhals-dev/narwhals) aims to support more functionnality from polars AND all of those from duckdb, as pql is not limited by multiple backend compatibility.
Furthermore, `narwhals` is primarly designed for library developpers who want integration with multiple dataframe libraries, not for end users.
Narwhals support a **subset** of polars API, hence necessarily a **subset** of DuckDB API, while pql aims to support the **full** API of both.

### SQLFrame

[SQLFrame](https://github.com/eakmanrq/sqlframe) is fundamentally a PySpark oriented library API-wise.

### Ibis

Ibis has a different syntax from polars. It can be close for some operations, but totally different for others.
Also the goal isn't the same, as Ibis is more focused on providing a high level API for multiple backends, while pql is focused on providing a polars like API for DuckDB.

## Architecture

The goal is to use as much as possible of the existing [Expression](https://duckdb.org/docs/stable/clients/python/expression) and [Relational](https://duckdb.org/docs/stable/clients/python/relational_api) API's.
raw SQL is only used when no other choice exist.

The `sql` folder is the internal API to create duckdb Expressions.
It handles data conversions, chainable API wrapping, and datatypes parsing and typing.
Then it is used in the top level modules who wraps this sql API to provide the user-facing API.

## Scripts

Scripts are used for code generation and API comparison at dev time.
They are not meant to be used by end users, and are not part of the public API.

More infos with the following command:

```shell
uv run -m scripts --help
```

### Comparator

The **compare** command will create the [coverage](API_COVERAGE.md) report to compare `pql` vs `polar`s and `narwhals` API's.

### Generators

The **gen-{fns, core, themes}** commands will respectively generate python code for:

- [The functions from the `table_functions` DuckDB table](src/pql/sql/_code_gen/_fns.py)
- [A core API wrapper around `DuckDBPyRelation` and `Expression`](src/pql/sql/_code_gen/_core.py)
- [A `Literal` for SQL display theming](src/pql/_typing.py) (see `Theme` type)

## References

- [DuckDB functions](https://duckdb.org/docs/stable/sql/functions/overview)
