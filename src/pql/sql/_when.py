from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import duckdb

from ._core import into_duckdb
from ._expr import SqlExpr
from ._funcs import reduce
from .utils import try_chain

if TYPE_CHECKING:
    from .typing import IntoExpr
    from .utils import TryIter

_red_fn = partial(reduce, function=SqlExpr.and_)


def when(predicates: TryIter[IntoExpr], *more_predicates: IntoExpr) -> When:
    return _red_fn(try_chain(predicates, more_predicates)).pipe(When)


@dataclass(slots=True)
class When:
    _when: IntoExpr

    def then(self, value: IntoExpr) -> Then:
        """Attach the value for the initial WHEN condition."""
        return Then(duckdb.CaseExpression(into_duckdb(self._when), into_duckdb(value)))


@dataclass(slots=True)
class Then(SqlExpr):
    def when(
        self, predicates: TryIter[IntoExpr], *more_predicates: IntoExpr
    ) -> ChainedWhen:
        return ChainedWhen(self, _red_fn(try_chain(predicates, more_predicates)))

    def otherwise(self, statement: IntoExpr) -> SqlExpr:
        return SqlExpr(self.inner().otherwise(into_duckdb(statement)))


@dataclass(slots=True)
class ChainedWhen:
    _chained_when: SqlExpr
    _predicate: IntoExpr

    def then(self, statement: IntoExpr) -> ChainedThen:
        return ChainedThen(
            self._chained_when.inner().when(
                into_duckdb(self._predicate), into_duckdb(statement)
            )
        )


@dataclass(slots=True)
class ChainedThen(SqlExpr):
    def when(
        self, predicates: TryIter[IntoExpr], *more_predicates: IntoExpr
    ) -> ChainedWhen:
        return ChainedWhen(self, _red_fn(try_chain(predicates, more_predicates)))

    def otherwise(self, statement: IntoExpr) -> SqlExpr:
        return SqlExpr(self.inner().otherwise(into_duckdb(statement)))
