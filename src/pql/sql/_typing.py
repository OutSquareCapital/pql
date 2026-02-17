from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import duckdb
import polars as pl
from polars._typing import FrameInitTypes

if TYPE_CHECKING:
    from .._expr import Expr
    from ._expr import SqlExpr


type FrameInit = (
    duckdb.DuckDBPyRelation | str | pl.DataFrame | pl.LazyFrame | FrameInitTypes
)
"""Inputs that can initialize a `LazyFrame`."""
type NumericLiteral = int | float | Decimal
type TemporalLiteral = date | time | datetime | timedelta

type NonNestedLiteral = (
    NumericLiteral | TemporalLiteral | str | bool | bytes | bytearray | memoryview
)
type PythonLiteral = (
    NonNestedLiteral | Sequence[PythonLiteral] | dict[Any, PythonLiteral] | None
)
"""Python literal types (can convert into a `lit` expression)."""
type ExprLike = SqlExpr | Expr
"""Types that are already expressions wrappers and can be used directly as expressions."""
type IntoExprColumn = str | ExprLike
"""Inputs that can convert into a `col` expression."""
type IntoExpr = PythonLiteral | IntoExprColumn
"""Inputs that can convert into an expression (either a `lit` or a `col`)."""
