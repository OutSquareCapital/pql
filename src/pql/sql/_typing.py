from collections.abc import Callable, Iterable
from datetime import date, datetime, time, timedelta
from typing import Any, Concatenate, Protocol, Self

import duckdb
import polars as pl
from polars._typing import FrameInitTypes


class ExprLike[T](Protocol):
    def pipe[**P, R](
        self,
        function: Callable[Concatenate[Self, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R: ...

    def _new(self, expr: T) -> T: ...

    def inner(self) -> T: ...


type FrameInit = duckdb.DuckDBPyRelation | pl.DataFrame | pl.LazyFrame | FrameInitTypes
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
type IntoExpr = PyLiteral | ExprLike[Any] | duckdb.Expression
type IntoExprColumn = Iterable[ExprLike[Any]] | ExprLike[Any] | str | duckdb.Expression
