# Roadmap

Here are notes of ideas and to-dos.

## TBD (to check, or verify)

- Views, etc... that are duckdb AND relation/expression specific, have to check more.
- Check how to integrate with narwhals (for framelib)
- There's probably methods on operators that are duplicated with the `SqlExpr` methods.
- Analyze geometric datatypes.
- Full interop with data formats and network(iceberg, credentials, etc...), for polars parity and full duckdb support.

## Refactors

- Metadata handling is messy AF

## Features

- Support geometrics datatypes (after check mentionned above)
- Support ALL duckdb Expr funcs that are not supported by polars (and hence don't appear in [the coverage](API_COVERAGE.md))

- Check the interop between polars SQL interface and duckdb SQL interface, and see if we can make them work together in a way that is more seamless than the current state of affairs.
- Same but for expressions. potentially improve the current pushdowns of polars lazy exprs with duckdb exprs.
