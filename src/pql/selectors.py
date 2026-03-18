"""Column selectors for PQL."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Self, overload, override

import pyochain as pc

from . import _datatypes as dt  # pyright: ignore[reportPrivateUsage]
from ._expr import Expr
from ._meta import Marker, MultiMeta, Resolver, ResolverFn

if TYPE_CHECKING:
    from .sql.typing import IntoExpr


class Selector(Expr):
    """Column selector based on dtype predicates."""

    meta: MultiMeta  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def _resolver(self) -> ResolverFn:
        return self.meta.resolver

    @classmethod
    def __from_resolver__(cls, resolver: ResolverFn) -> Self:
        return cls(Marker.MULTI.to_expr(), MultiMeta(Marker.MULTI, resolver=resolver))

    @overload
    def union(self, other: Self) -> Self: ...
    @overload
    def union(self, other: IntoExpr) -> Expr: ...
    def union(self, other: IntoExpr) -> Self | Expr:
        match other:
            case Selector():
                return self.__from_resolver__(
                    Resolver.union(self._resolver, other._resolver)
                )
            case _:
                return Expr.__or__(self, other)

    @overload
    def __or__(self, other: Self) -> Self: ...
    @overload
    def __or__(self, other: IntoExpr) -> Expr: ...
    @override
    def __or__(self, other: IntoExpr) -> Self | Expr:
        return self.union(other)

    @overload
    def intersection(self, other: Self) -> Self: ...
    @overload
    def intersection(self, other: IntoExpr) -> Expr: ...
    def intersection(self, other: IntoExpr) -> Self | Expr:
        match other:
            case Selector():
                return self.__from_resolver__(
                    Resolver.intersection(self._resolver, other._resolver)
                )
            case _:
                return Expr.__and__(self, other)

    @overload
    def __and__(self, other: Self) -> Self: ...
    @overload
    def __and__(self, other: IntoExpr) -> Expr: ...
    @override
    def __and__(self, other: IntoExpr) -> Self | Expr:
        return self.intersection(other)

    @overload
    def difference(self, other: Self) -> Self: ...
    @overload
    def difference(self, other: IntoExpr) -> Expr: ...
    def difference(self, other: IntoExpr) -> Self | Expr:
        match other:
            case Selector():
                return self.__from_resolver__(
                    Resolver.difference(self._resolver, other._resolver)
                )
            case _:
                return Expr.__sub__(self, other)

    @overload
    def __sub__(self, other: Self) -> Self: ...
    @overload
    def __sub__(self, other: IntoExpr) -> Expr: ...
    @override
    def __sub__(self, other: IntoExpr) -> Self | Expr:
        return self.difference(other)

    def complement(self) -> Selector:
        return self.__from_resolver__(Resolver.complement(self._resolver))

    @override
    def __invert__(self) -> Selector:
        return self.complement()


def by_dtype(*dtypes: type[dt.DataType]) -> Selector:
    """Select columns matching any of the given dtype classes."""
    return Selector.__from_resolver__(Resolver.dtype(*dtypes))


def numeric() -> Selector:
    """Select all numeric columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.NumericType))


def string() -> Selector:
    """Select all string columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.StringType))


def boolean() -> Selector:
    """Select all boolean columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.Boolean))


def all() -> Selector:
    """Select all columns."""
    return Selector.__from_resolver__(Resolver.all_columns())


def float() -> Selector:
    """Select all float columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.FloatType))


def integer() -> Selector:
    """Select all integer columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.IntegerType))


def signed_integer() -> Selector:
    """Select all signed integer columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.SignedIntegerType))


def unsigned_integer() -> Selector:
    """Select all unsigned integer columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.UnsignedIntegerType))


def temporal() -> Selector:
    """Select all temporal columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.TemporalType))


def date() -> Selector:
    """Select all date columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.Date))


def time() -> Selector:
    """Select all time columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.Time))


def duration() -> Selector:
    """Select all duration columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.Duration))


def binary() -> Selector:
    """Select all binary columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.Binary))


def enum() -> Selector:
    """Select all enum columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.Enum))


def decimal() -> Selector:
    """Select all decimal columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.Decimal))


def nested() -> Selector:
    """Select all nested (list, array, struct, map) columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.NestedType))


def struct() -> Selector:
    """Select all struct columns."""
    return Selector.__from_resolver__(Resolver.dtype(dt.Struct))


# ──── name-based selectors ────


def matches(pattern: str) -> Selector:
    """Select columns whose names match the given regex pattern."""
    compiled = re.compile(pattern)
    return Selector.__from_resolver__(
        Resolver.name(lambda name: compiled.search(name) is not None)
    )


def by_name(*names: str) -> Selector:
    """Select columns by exact name."""
    return Selector.__from_resolver__(Resolver.ordered_name(pc.Seq(names)))


def starts_with(*prefix: str) -> Selector:
    """Select columns whose names start with any of the given prefixes."""
    return Selector.__from_resolver__(
        Resolver.name(lambda name: name.startswith(prefix))
    )


def ends_with(*suffix: str) -> Selector:
    """Select columns whose names end with any of the given suffixes."""
    return Selector.__from_resolver__(Resolver.name(lambda name: name.endswith(suffix)))


def contains(*substring: str) -> Selector:
    """Select columns whose names contain any of the given substrings."""
    subs = pc.Seq(substring)
    return Selector.__from_resolver__(
        Resolver.name(lambda name: subs.iter().any(lambda s: s in name))
    )


__all__ = [
    "all",
    "binary",
    "boolean",
    "by_dtype",
    "by_name",
    "contains",
    "date",
    "decimal",
    "duration",
    "ends_with",
    "enum",
    "float",
    "integer",
    "matches",
    "nested",
    "numeric",
    "signed_integer",
    "starts_with",
    "string",
    "struct",
    "temporal",
    "time",
    "unsigned_integer",
]
