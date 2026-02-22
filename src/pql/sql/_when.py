from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb

from ._expr import SqlExpr
from ._funcs import into_expr

if TYPE_CHECKING:
    from .typing import IntoExpr


def when(condition: IntoExpr) -> When:
    return When(condition)


@dataclass(slots=True)
class When:
    _when: IntoExpr

    def then(self, value: IntoExpr) -> Then:
        """Attach the value for the initial WHEN condition."""
        return Then(
            duckdb.CaseExpression(
                into_expr(self._when, as_col=True).inner(),
                into_expr(value, as_col=True).inner(),
            )
        )


@dataclass(slots=True)
class Then(SqlExpr):
    def when(self, predicate: IntoExpr) -> ChainedWhen:
        return ChainedWhen(self, into_expr(predicate, as_col=True))

    def otherwise(self, statement: IntoExpr) -> SqlExpr:
        return SqlExpr(
            self.inner().otherwise(into_expr(statement, as_col=True).inner())
        )


@dataclass(slots=True)
class ChainedWhen:
    _chained_when: SqlExpr
    _predicate: SqlExpr

    def then(self, statement: IntoExpr) -> ChainedThen:
        return ChainedThen(
            self._chained_when.inner().when(
                self._predicate.inner(), into_expr(statement, as_col=True).inner()
            )
        )


@dataclass(slots=True)
class ChainedThen(SqlExpr):
    def when(self, predicate: IntoExpr) -> ChainedWhen:
        return ChainedWhen(self, into_expr(predicate, as_col=True))

    def otherwise(self, statement: IntoExpr) -> SqlExpr:
        return SqlExpr(
            self.inner().otherwise(into_expr(statement, as_col=True).inner())
        )
