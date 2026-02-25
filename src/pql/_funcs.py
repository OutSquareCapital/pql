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
    return Expr(inner, pc.Some(ExprMeta(inner.inner().get_name(), is_multi=True)))


_ELEMENT = Expr(sql.element())


def element() -> Expr:
    """Alias for an element being evaluated in a list context."""
    return _ELEMENT
