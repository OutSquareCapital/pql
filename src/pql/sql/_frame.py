from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import pyochain as pc
from sqlglot import exp

from ._core import CoreHandler
from ._funcs import into_expr
from .utils import TryIter, try_iter

if TYPE_CHECKING:
    from pyochain.traits import PyoIterable

    from .typing import IntoExpr, JoinStrategy, PythonLiteral


class Frame(CoreHandler[exp.Select]):
    _inner: exp.Select
    __slots__: ClassVar[Iterable[str]] = ("_inner",)

    def __init__(self, expr: exp.Select | None = None) -> None:
        self._inner = expr or exp.select(dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]

    def select(self, columns: Iterable[IntoExpr]) -> Self:
        cols = pc.Iter(columns).map(lambda e: into_expr(e, as_col=True).inner())
        qry = self.inner().select(*cols, dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]
        return self._new(qry)

    def filter(self, conditions: Iterable[IntoExpr]) -> Self:
        qry = self.inner().where(  # pyright: ignore[reportUnknownMemberType]
            *pc.Iter(conditions).map(lambda e: into_expr(e, as_col=True).inner()),
            dialect="duckdb",
        )
        return self._new(qry)

    def aggregate(self, aggregations: Iterable[IntoExpr]) -> Self:
        aggs = pc.Iter(aggregations).map(lambda e: into_expr(e, as_col=True).inner())
        qry = self.inner().group_by(*aggs, dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]
        return self._new(qry)

    def limit(self, n: int, offset: int | None = None) -> Self:
        qry = self.inner().limit(n, offset=offset, dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]
        return self._new(qry)

    def join_asof(
        self,
        other: Self,
        condition: IntoExpr,
        select_cols: PyoIterable[str],
        how: Literal["left", "inner"],
    ) -> Self:
        join_type = "asof left" if how == "left" else "asof"
        qry = (
            self.select(select_cols)
            .inner()
            .join(  # pyright: ignore[reportUnknownMemberType]
                other.inner(), on=into_expr(condition).to_sql(), join_type=join_type
            )
        )

        return self._new(qry)

    def pivot(
        self,
        on: TryIter[str],
        using: TryIter[str],
        group_by: TryIter[str],
        in_values: pc.Option[Sequence[PythonLiteral]],
        order_by: TryIter[str],
    ) -> Self:

        def _on_exprs(on_iter: pc.Seq[str]) -> PyoIterable[exp.Expr] | PyoIterable[str]:
            match in_values:
                case pc.Some(vals):
                    converted = pc.Iter(vals).map(exp.convert).collect()
                    expr = exp.In(this=on_iter.first(), expressions=converted)
                    return pc.Iter.once(expr)
                case _:
                    return on_iter

        def _group() -> exp.Group | None:
            group = try_iter(group_by).then(
                lambda cols: exp.Group(expressions=cols.collect(list))
            )
            return group.unwrap() if group.is_some() else None

        def _pivot() -> exp.Pivot:
            return exp.Pivot(
                this=self.inner(),
                expressions=try_iter(on).collect().into(_on_exprs),
                using=try_iter(using),
                group=_group(),
            )

        def _select_ordered(cols: Iterable[str]) -> exp.Select:
            return (
                exp.select("*")  # pyright: ignore[reportUnknownMemberType]
                .from_(exp.Subquery(this=_pivot()))
                .order_by(*cols)
            )

        qry = try_iter(order_by).collect().then(_select_ordered).unwrap_or_else(_pivot)

        return self._new(qry)

    def unpivot(
        self,
        on: TryIter[str],
        index: TryIter[str],
        variable_name: str,
        value_name: str,
        order_by: TryIter[str] = None,
    ) -> Self:
        """Unpivot from wide to long format."""
        index_cols = try_iter(index).collect(dict.fromkeys)
        unpivot_cols = (
            try_iter(on)
            .then_some()
            .unwrap_or_else(
                lambda: self.columns.iter().filter(lambda name: name not in index_cols)
            )
        )

        def _unpivot() -> exp.Pivot:
            return exp.Pivot(
                this=self.inner(),
                expressions=unpivot_cols,
                unpivot=True,
                into=exp.UnpivotColumns(this=variable_name, expressions=(value_name,)),
            )

        def _select() -> exp.Select:
            sub_qry = exp.Subquery(this=_unpivot())
            return exp.select(*index_cols, variable_name, value_name).from_(sub_qry)  # pyright: ignore[reportUnknownMemberType]

        qry = (
            try_iter(order_by)
            .then(lambda cols: _select().order_by(*cols))  # pyright: ignore[reportUnknownMemberType]
            .unwrap_or_else(_select)
        )

        return self._new(qry)

    def distinct(self) -> Self:
        qry = self.inner().distinct()
        return self._new(qry)

    def union(self, other: Self) -> Self:
        qry = self.inner().union(other.inner(), dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]
        return self._new(qry)

    def join(
        self,
        other: Self,
        on: IntoExpr = None,
        condition: IntoExpr = None,
        how: JoinStrategy = "inner",
    ) -> Self:
        qry = self.inner().join(  # pyright: ignore[reportUnknownMemberType]
            other.inner(),
            using=into_expr(condition).inner() or None,
            on=into_expr(on).inner() or None,
            join_type=how,
            dialect="duckdb",
        )
        return self._new(qry)

    def cross(self, other: Self) -> Self:
        qry = self.inner().join(other.inner(), join_type="cross", dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]
        return self._new(qry)

    def sort(self, by: Iterable[IntoExpr]) -> Self:
        qry = self.inner().order_by(  # pyright: ignore[reportUnknownMemberType]
            *pc.Iter(by).map(lambda e: into_expr(e, as_col=True).inner()),
            dialect="duckdb",
        )
        return self._new(qry)

    def sql_query(self) -> str:
        """Return the SQL query string for the current frame."""
        return self.inner().sql(dialect="duckdb")
