from collections.abc import Iterable

from . import sql
from ._expr import Expr
from .sql._typing import IntoExpr, IntoExprColumn


class Col:
    def __call__(self, name: str) -> Expr:
        return Expr(sql.col(name))

    def __getattr__(self, name: str) -> Expr:
        return self.__call__(name)


def lit(value: IntoExpr) -> Expr:
    """Create a literal expression."""
    return Expr(sql.lit(value))


col: Col = Col()


def all(exclude: Iterable[IntoExprColumn] | None = None) -> Expr:
    """Create an expression representing all columns (equivalent to pl.all())."""
    return Expr(sql.all(exclude))


_ELEMENT = Expr(sql.element())


def element() -> Expr:
    """Alias for an element being evaluated in a list context."""
    return _ELEMENT
