"""SQL function namespaces for SQL expressions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from pql.sql.typing import IntoExprColumn

from ._code_gen import (
    ArrayFns,
    DateTimeFns,
    EnumFns,
    GeoSpatialFns,
    JsonFns,
    ListFns,
    MapFns,
    RegexFns,
    StringFns,
    StructFns,
)
from ._core import func
from ._expr import SqlExpr

if TYPE_CHECKING:
    from .typing import IntoExpr, IntoExprColumn


@dataclass(slots=True)
class SqlExprStringNameSpace(StringFns[SqlExpr]):
    """String function namespace for SQL expressions."""

    def strftime(self, format_arg: IntoExprColumn | date | datetime | str) -> SqlExpr:
        """Converts a `date` to a string according to the format string.

        **SQL name**: *strftime*

        Args:
            format_arg (IntoExprColumn | date | datetime | str): `DATE | TIMESTAMP | TIMESTAMP_NS | VARCHAR` expression

        Returns:
            SqlExpr
        """
        return self._new(func("strftime", self.inner(), format_arg))

    def strptime(self, format_arg: IntoExprColumn | list[str]) -> SqlExpr:
        """Converts the `string` text to timestamp according to the format string.

        Throws an error on failure.

        To return `NULL` on failure, use try_strptime.

        **SQL name**: *strptime*

        Args:
            format_arg (IntoExprColumn | list[str]): `VARCHAR | VARCHAR[]` expression

        Returns:
            SqlExpr
        """
        return self._new(func("strptime", self.inner(), format_arg))

    def concat(self, *args: IntoExpr) -> SqlExpr:
        """Concatenates multiple strings or lists.

        `NULL` inputs are skipped.

        See also operator `||`.

        **SQL name**: *concat*

        Args:
            *args (IntoExpr): `ANY` expression

        Returns:
            SqlExpr
        """
        return self._new(func("concat", self.inner(), *args))


@dataclass(slots=True)
class SqlExprListNameSpace(ListFns[SqlExpr]):
    """List function namespace for SQL expressions."""

    def eval(self, expr: SqlExpr) -> SqlExpr:
        """Run an expression against each array element."""
        from ._funcs import fn_once

        return self._new(self.transform(fn_once(expr.inner())).inner())

    def std(self, ddof: int = 1) -> SqlExpr:
        """Compute the standard deviation of the lists in the column."""
        match ddof:
            case 0:
                return self.stddev_pop()
            case _:
                return self.stddev_samp()

    def var(self, ddof: int = 1) -> SqlExpr:
        """Compute the variance of the lists in the column."""
        match ddof:
            case 0:
                return self.var_pop()
            case _:
                return self.var_samp()

    def filter(self, lambda_arg: IntoExprColumn) -> SqlExpr:
        """Constructs a list from those elements of the input `list` for which the `lambda` function returns `true`.

        DuckDB must be able to cast the `lambda` function's return type to `BOOL`.

        The return type of `list_filter` is the same as the input list's.

        **SQL name**: *filter*

        Args:
            lambda_arg (IntoExprColumn): `LAMBDA` expression

        Examples:
            filter([3, 4, 5], lambda x : x > 4)

        Returns:
            T
        """
        from ._funcs import fn_once

        return self._new(func("list_filter", self.inner(), fn_once(lambda_arg)))


@dataclass(slots=True)
class SqlExprStructNameSpace(StructFns[SqlExpr]):
    """Struct function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprDateTimeNameSpace(DateTimeFns[SqlExpr]):
    """Datetime function namespace for SQL expressions."""

    def trunc(self, precision: IntoExprColumn) -> SqlExpr:
        """Truncate to specified precision.

        **SQL name**: *date_trunc*

        Args:
            precision (IntoExprColumn): `VARCHAR` expression

        Examples:
            date_trunc('hour', TIMESTAMPTZ '1992-09-20 20:38:40')

        Returns:
            T
        """
        return self._new(func("date_trunc", precision, self.inner()))


@dataclass(slots=True)
class SqlExprArrayNameSpace(ArrayFns[SqlExpr]):
    """Array function namespace for SQL expressions."""

    def eval(self, expr: SqlExpr) -> SqlExpr:
        """Run an expression against each array element."""
        from ._funcs import fn_once

        return self._new(self.transform(fn_once(expr.inner())).inner())

    def filter(self, lambda_arg: IntoExprColumn) -> SqlExpr:
        """Constructs a list from those elements of the input `list` for which the `lambda` function returns `true`.

        DuckDB must be able to cast the `lambda` function's return type to `BOOL`.

        The return type of `list_filter` is the same as the input list's.

        **SQL name**: *filter*

        Args:
            lambda_arg (IntoExprColumn): `LAMBDA` expression

        Examples:
            filter([3, 4, 5], lambda x : x > 4)

        Returns:
            T
        """
        from ._funcs import fn_once

        return self._new(func("array_filter", self.inner(), fn_once(lambda_arg)))


@dataclass(slots=True)
class SqlExprJsonNameSpace(JsonFns[SqlExpr]):
    """JSON function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprRegexNameSpace(RegexFns[SqlExpr]):
    """Regex function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprMapNameSpace(MapFns[SqlExpr]):
    """Map function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprEnumNameSpace(EnumFns[SqlExpr]):
    """Enum function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprGeoSpatialNameSpace(GeoSpatialFns[SqlExpr]):
    """Geospatial function namespace for SQL expressions."""
