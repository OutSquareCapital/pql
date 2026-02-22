from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import sql
from ._expr import Expr

if TYPE_CHECKING:
    from .sql.typing import IntoExpr


def when(condition: IntoExpr) -> When:
    return When(sql.when(condition))


@dataclass(slots=True)
class When:
    _when: sql.When

    def then(self, value: IntoExpr) -> Then:
        return Then(self._when.then(value))


@dataclass(slots=True)
class Then(Expr):
    _inner: sql.Then  # pyright: ignore[reportIncompatibleVariableOverride]

    def when(self, predicate: IntoExpr) -> ChainedWhen:
        return ChainedWhen(self._inner.when(predicate))

    def otherwise(self, statement: IntoExpr) -> Expr:
        return Expr(self._inner.otherwise(statement))


@dataclass(slots=True)
class ChainedWhen:
    _chained_when: sql.ChainedWhen

    def then(self, statement: IntoExpr) -> ChainedThen:
        return ChainedThen(self._chained_when.then(statement))


@dataclass(slots=True)
class ChainedThen(Expr):
    _inner: sql.ChainedThen  # pyright: ignore[reportIncompatibleVariableOverride]

    def when(self, predicate: IntoExpr) -> ChainedWhen:
        return ChainedWhen(self._inner.when(predicate))

    def otherwise(self, statement: IntoExpr) -> Expr:
        return Expr(self._inner.otherwise(statement))
