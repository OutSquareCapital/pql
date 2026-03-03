from collections.abc import Iterable

import pyochain as pc

from . import sql
from ._expr import Expr, ExprMeta
from .sql.typing import IntoExprColumn, PythonLiteral


class Col:
    def __call__(self, name: str) -> Expr:
        return Expr(sql.col(name))

    def __getattr__(self, name: str) -> Expr:
        return self.__call__(name)


def lit(value: PythonLiteral) -> Expr:
    """Create a literal expression."""
    return Expr(sql.lit(value))


col: Col = Col()


def all(exclude: Iterable[IntoExprColumn] | None = None) -> Expr:
    """Create an expression representing all columns (equivalent to pl.all())."""
    inner = sql.all(exclude)
    excluded_names = (
        pc.Option(exclude)
        .map(
            lambda exc: (
                pc.Iter(exc)
                .map(lambda value: sql.into_expr(value, as_col=True).get_name())
                .collect(pc.Set)
            )
        )
        .unwrap_or_else(pc.Set[str].new)
    )
    meta = pc.Some(
        ExprMeta(inner.get_name(), is_multi=True, excluded_names=excluded_names)
    )
    return Expr(inner, meta)


_ELEMENT = Expr(sql.element())


def element() -> Expr:
    """Alias for an element being evaluated in a list context."""
    return _ELEMENT
