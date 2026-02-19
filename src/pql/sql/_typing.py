from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import duckdb
from narwhals.typing import IntoFrame

if TYPE_CHECKING:
    from .._expr import Expr
    from ._core import DuckHandler
    from ._expr import SqlExpr

type ExprLike = SqlExpr | Expr
"""Types that are already expressions wrappers and can be used directly as expressions."""
type IntoValues = DuckHandler | Mapping[str, Any] | list[Any] | tuple[duckdb.Expression]
"""Types that can be converted into a `values` relation (either an expression, a mapping, or a sequence)."""
type FrameInit = duckdb.DuckDBPyRelation | str | IntoFrame | IntoValues
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
type IntoExprColumn = str | ExprLike
"""Inputs that can convert into a `col` expression."""
type IntoExpr = PythonLiteral | IntoExprColumn
"""Inputs that can convert into an expression (either a `lit` or a `col`)."""
