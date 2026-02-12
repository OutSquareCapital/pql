from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

import duckdb
import polars as pl
from polars._typing import FrameInitTypes

if TYPE_CHECKING:
    from .._expr import Expr
    from ._expr import SqlExpr

type ExprLike = SqlExpr | Expr | duckdb.Expression


type FrameInit = (
    duckdb.DuckDBPyRelation | str | pl.DataFrame | pl.LazyFrame | FrameInitTypes
)
type PyLiteral = (
    str
    | int
    | float
    | bool
    | date
    | datetime
    | time
    | timedelta
    | bytes
    | bytearray
    | memoryview
    | list[PyLiteral]
    | dict[Any, PyLiteral]
    | None
)
type IntoExpr = PyLiteral | ExprLike
type IntoExprColumn = str | ExprLike | Iterable[ExprLike]
