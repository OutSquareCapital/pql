from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb

from ._expr import SqlExpr
from ._funcs import into_duckdb

if TYPE_CHECKING:
    from .typing import IntoExpr


def when(condition: IntoExpr) -> When:
    return When(condition)


@dataclass(slots=True)
class When:
    _when: IntoExpr

    def then(self, value: IntoExpr) -> Then:
        """Attach the value for the initial WHEN condition."""
        return Then(duckdb.CaseExpression(into_duckdb(self._when), into_duckdb(value)))


@dataclass(slots=True)
class Then(SqlExpr):
    def when(self, predicate: IntoExpr) -> ChainedWhen:
        return ChainedWhen(self, predicate)

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
    def when(self, predicate: IntoExpr) -> ChainedWhen:
        return ChainedWhen(self, predicate)

    def otherwise(self, statement: IntoExpr) -> SqlExpr:
        return SqlExpr(self.inner().otherwise(into_duckdb(statement)))
