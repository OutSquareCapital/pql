"""Typing definitions for the SQL module."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import duckdb
from narwhals.typing import IntoFrame

if TYPE_CHECKING:
    from _duckdb._typing import (  # pyright: ignore[reportMissingModuleSource]
        BlobLiteral as DuckBlobLit,
        IntoExpr as DuckIntoExpr,
        IntoExprColumn as DuckIntoExprColumn,
        NestedLiteral as DuckNestedLit,
        NonNestedLiteral as DuckNonNestedLit,
        PythonLiteral as DuckPyLit,
        PyTypeIds as DuckPyTypeIds,
        StrIntoPyType as DuckStrIntoPyType,
    )

    from .._expr import Expr
    from ._core import DuckHandler
    from ._expr import SqlExpr


@runtime_checkable
class FrameLike(Protocol):
    """Protocol to check if a type is indeed a narwhals `IntoFrame`."""

    @property
    def columns(self) -> Any: ...  # noqa: ANN401, D102


class NPProtocol(Protocol):
    """Base Protocol for numpy objects."""

    @property
    def dtype(self) -> Any: ...  # noqa: ANN401, D102
    @property
    def ndim(self) -> int: ...  # noqa: D102
    def __array__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401, D105
    def __array_wrap__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401, D105
    @property
    def __array_interface__(self) -> dict[str, Any]: ...  # noqa: D105
    @property
    def __array_priority__(self) -> float: ...  # noqa: D105


class NPScalarTypeLike(NPProtocol, Protocol):  # noqa: D101
    @property
    def itemsize(self) -> int: ...  # noqa: D102


@runtime_checkable
class NPArrayLike[S: tuple[Any, ...], D](NPProtocol, Protocol):
    """Protocol for `numpy` ndarrays."""

    def __len__(self) -> int: ...  # noqa: D105
    def __contains__(self, value: object, /) -> bool: ...  # noqa: D105
    def __iter__(self) -> Iterator[D]: ...  # noqa: D105
    def __array_finalize__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: ANN401, D105
    def __getitem__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401, D105
    def __setitem__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: ANN401, D105
    @property
    def shape(self) -> S: ...  # noqa: D102
    @property
    def size(self) -> int: ...  # noqa: D102


type IntoDict[K, V] = Mapping[K, V] | Iterable[tuple[K, V]]
type ExprLike = SqlExpr | Expr | DuckHandler
"""Types that are already expressions wrappers and can be used directly as expressions."""
type BlobLiteral = DuckBlobLit
type NonNestedLiteral = DuckNonNestedLit
type SeqLiteral[T: NonNestedLiteral] = list[T] | tuple[T, ...]
"""Sequence of non-nested literals of the same type."""
type PythonLiteral = DuckPyLit
type NestedLiteral = DuckNestedLit
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
type IntoExpr = PythonLiteral | IntoExprColumn | duckdb.Expression
"""Inputs that can convert into an expression (either a `lit` or a `col`)."""
type IntoDuckExpr = DuckIntoExpr
type IntoDuckExprCol = DuckIntoExprColumn
type DTypeIds = DuckPyTypeIds
type StrIntoDType = DuckStrIntoPyType
