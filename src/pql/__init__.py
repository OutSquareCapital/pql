"""pql - Polars-like SQL Generator for DuckDB.

Write Polars-like code, generate DuckDB SQL.

Example:
-------
>>> import pql
>>> query = (
...     pql.scan("my_table")
...     .filter(pql.col("age") > 18)
...     .with_columns(
...         (pql.col("salary") * 1.1).alias("adjusted_salary"),
...     )
...     .group_by("department")
...     .agg(pql.col("adjusted_salary").mean().alias("avg_salary"))
... )
>>> print(query.sql(pretty=True))
"""

from __future__ import annotations

from .datatypes import DataType
from .expr import Expr
from .functions import (
    abs,
    all,
    arr_contains,
    arr_get,
    arr_length,
    avg,
    cast,
    ceil,
    coalesce,
    col,
    cols,
    concat,
    concat_str,
    count,
    date,
    datetime,
    dense_rank,
    exclude,
    explode,
    fill_null,
    first,
    floor,
    format,
    implode,
    is_not_null,
    is_null,
    lag,
    last,
    lead,
    lit,
    log,
    log10,
    max,
    mean,
    median,
    min,
    n_unique,
    now,
    pql_exp,
    quantile,
    rank,
    round,
    row_number,
    sqrt,
    std,
    struct,
    sum,
    today,
    var,
    when,
)
from .lazyframe import GroupBy, LazyFrame, scan

__all__ = [
    "DataType",
    # Core classes
    "Expr",
    "GroupBy",
    "LazyFrame",
    # Math
    "abs",
    "all",
    "arr_contains",
    "arr_get",
    # Array
    "arr_length",
    "avg",
    # Type casting
    "cast",
    "ceil",
    "coalesce",
    "col",
    "cols",
    # String
    "concat",
    "concat_str",
    "count",
    # Date/Time
    "date",
    "datetime",
    "dense_rank",
    "exclude",
    "explode",
    "fill_null",
    "first",
    "floor",
    "format",
    "implode",
    "is_not_null",
    # Null handling
    "is_null",
    "lag",
    "last",
    "lead",
    "lit",
    "log",
    "log10",
    "max",
    "mean",
    "median",
    "min",
    "n_unique",
    "now",
    "pql_exp",
    "quantile",
    # Window
    "rank",
    "round",
    "row_number",
    # Constructors
    "scan",
    "sqrt",
    "std",
    # Struct
    "struct",
    # Aggregations
    "sum",
    "today",
    "var",
    # Conditional
    "when",
]

__version__ = "0.1.0"
