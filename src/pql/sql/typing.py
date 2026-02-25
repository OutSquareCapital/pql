"""Typing definitions for the SQL module."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

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
    def columns(self) -> Any: ...  # noqa: ANN401, D102


@runtime_checkable
class NPArrayLike[S, D](Protocol):
    """Protocol for `numpy` ndarrays."""

    def __len__(self) -> int: ...  # noqa: D105
    def __contains__(self, value: object, /) -> bool: ...  # noqa: D105

    def __iter__(self) -> Iterator[D]: ...  # noqa: D105
    def __array__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401, D105
    def __array_finalize__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: ANN401, D105
    def __array_wrap__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401, D105
    def __getitem__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401, D105
    def __setitem__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: ANN401, D105
    @property
    def shape(self) -> S: ...  # noqa: D102
    @property
    def dtype(self) -> Any: ...  # noqa: ANN401, D102
    @property
    def ndim(self) -> int: ...  # noqa: D102
    @property
    def size(self) -> int: ...  # noqa: D102


type IntoDict[K, V] = Mapping[K, V] | Iterable[tuple[K, V]]
type ExprLike = SqlExpr | Expr
"""Types that are already expressions wrappers and can be used directly as expressions."""

"""Inputs that can initialize a `LazyFrame`."""
type NumericLiteral = int | float | Decimal
type TemporalLiteral = date | time | datetime | timedelta
type BlobLiteral = bytes | bytearray | memoryview
type NonNestedLiteral = (
    NumericLiteral | TemporalLiteral | str | bool | BlobLiteral | UUID
)
type NestedLiteral = (
    list[PythonLiteral]
    | dict[NonNestedLiteral, PythonLiteral]
    | tuple[PythonLiteral, ...]
    | NPArrayLike[Any, Any]
)
type PythonLiteral = NonNestedLiteral | NestedLiteral | None
"""Python literal types (can convert into a `lit` expression)."""
type ExprIntoVals = DuckHandler | duckdb.Expression
type SeqIntoVals = (
    Sequence[duckdb.Expression]
    | Sequence[Mapping[str, PythonLiteral]]
    | Sequence[PythonLiteral]
    | NPArrayLike[Any, Any]
)

type IntoValues = ExprIntoVals | Mapping[str, Sequence[PythonLiteral]] | SeqIntoVals
"""Types that can be converted into a `values` relation (either an expression, a mapping, or a sequence)."""
type IntoRel = IntoFrame | IntoValues
""""Types that can be converted into a relation (either a frame or values)."""
type IntoExprColumn = str | ExprLike
"""Inputs that can convert into a `col` expression."""
type IntoExpr = PythonLiteral | IntoExprColumn
"""Inputs that can convert into an expression (either a `lit` or a `col`)."""
