from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyochain as pc

from . import sql
from ._expr import Expr, ExprMeta

if TYPE_CHECKING:
    from .sql.typing import IntoExpr
    from .sql.utils import TryIter


def when(predicates: TryIter[IntoExpr], *more_predicates: IntoExpr) -> When:
    return When(sql.when(predicates, *more_predicates))


@dataclass(slots=True)
class When:
    _when: sql.When

    def then(self, value: IntoExpr) -> Then:
        return Then(self._when.then(value))


class Then(Expr):
    _inner: sql.Then
    __slots__ = ()

    def when(
        self, predicates: TryIter[IntoExpr], *more_predicates: IntoExpr
    ) -> ChainedWhen:
        return ChainedWhen(self._inner.when(predicates, *more_predicates), self.meta)

    def otherwise(self, statement: IntoExpr) -> Expr:
        return Expr(self._inner.otherwise(statement), pc.Some(self.meta))


@dataclass(slots=True)
class ChainedWhen:
    _chained_when: sql.ChainedWhen
    _meta: ExprMeta

    def then(self, statement: IntoExpr) -> ChainedThen:
        return ChainedThen(self._chained_when.then(statement), pc.Some(self._meta))


class ChainedThen(Expr):
    _inner: sql.ChainedThen
    __slots__ = ()

    def when(
        self, predicates: TryIter[IntoExpr], *more_predicates: IntoExpr
    ) -> ChainedWhen:
        return ChainedWhen(self._inner.when(predicates, *more_predicates), self.meta)

    def otherwise(self, statement: IntoExpr) -> Expr:
        return Expr(self._inner.otherwise(statement), pc.Some(self.meta))
