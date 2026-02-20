from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import duckdb
from narwhals.typing import IntoFrame

if TYPE_CHECKING:
    from .._expr import Expr
    from ._core import DuckHandler
    from ._expr import SqlExpr


@runtime_checkable
class FrameLike(Protocol):
    """Protocol to check if a type is indeed a narwhals `IntoFrame`."""

    @property
    def columns(self) -> Any: ...  # noqa: ANN401


@runtime_checkable
class NPArrayLike[S, D](Protocol):
    """Protocol for `numpy` ndarrays."""

    def __len__(self) -> int: ...
    def __contains__(self, value: object, /) -> bool: ...

    def __iter__(self) -> Iterator[D]: ...
    def __array__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401
    def __array_finalize__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: ANN401
    def __array_wrap__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401
    def __getitem__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401
    def __setitem__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: ANN401
    @property
    def shape(self) -> S: ...
    @property
    def dtype(self) -> Any: ...  # noqa: ANN401
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...


type ExprLike = SqlExpr | Expr
"""Types that are already expressions wrappers and can be used directly as expressions."""
type ExprIntoVals = DuckHandler | duckdb.Expression
type SeqIntoVals = (
    Sequence[duckdb.Expression]
    | Sequence[Mapping[str, PythonLiteral]]
    | Sequence[PythonLiteral]
)
type IntoValues = ExprIntoVals | Mapping[str, Sequence[PythonLiteral]] | SeqIntoVals
"""Types that can be converted into a `values` relation (either an expression, a mapping, or a sequence)."""
type IntoRel = IntoFrame | IntoValues | NPArrayLike[Any, Any]
"""Inputs that can initialize a `LazyFrame`."""
type NumericLiteral = int | float | Decimal
type TemporalLiteral = date | time | datetime | timedelta

type NonNestedLiteral = (
    NumericLiteral | TemporalLiteral | str | bool | bytes | bytearray | memoryview
)
type PythonLiteral = (
    NonNestedLiteral
    | Sequence[PythonLiteral]
    | dict[NonNestedLiteral, PythonLiteral]
    | None
)
"""Python literal types (can convert into a `lit` expression)."""
type IntoExprColumn = str | ExprLike
"""Inputs that can convert into a `col` expression."""
type IntoExpr = PythonLiteral | IntoExprColumn
"""Inputs that can convert into an expression (either a `lit` or a `col`)."""
