from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import pyochain as pc
from sqlglot import exp

from ._code_gen import Relation
from ._creation import from_query, into_relation
from ._funcs import into_expr
from .utils import TryIter, try_iter

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation
    from pyochain.traits import PyoIterable

    from .typing import IntoExpr, IntoRel, Orientation, PythonLiteral


class Frame(Relation):
    _inner: DuckDBPyRelation
    __slots__: ClassVar[Iterable[str]] = ("_inner",)

    def __init__(self, data: IntoRel, orient: Orientation = "col") -> None:
        self._inner = into_relation(data, orient=orient)

    def join_asof(
        self,
        other: IntoRel,
        condition: IntoExpr,
        select_cols: PyoIterable[str],
        how: Literal["left", "inner"],
    ) -> Self:
        join_type = "asof left" if how == "left" else "asof"
        qry = (
            exp.select(*select_cols)  # pyright: ignore[reportUnknownMemberType]
            .from_("lhs")
            .join("rhs", on=into_expr(condition).to_sql(), join_type=join_type)
            .sql(dialect="duckdb")
        )

        return self.__class__(from_query(qry, lhs=self.inner(), rhs=other))

    def pivot(
        self,
        on: TryIter[str],
        using: TryIter[str],
        group_by: TryIter[str],
        in_values: Sequence[PythonLiteral] | None,
        order_by: TryIter[str],
    ) -> Self:

        def _on_exprs(on_iter: pc.Seq[str]) -> PyoIterable[exp.Expr] | PyoIterable[str]:
            match in_values:
                case Sequence() as vals:
                    return pc.Iter.once(
                        exp.In(
                            this=on_iter.first(),
                            expressions=pc.Iter(vals).map(exp.convert).collect(),
                        )
                    )
                case None:
                    return on_iter

        def _group() -> exp.Group | None:
            group = try_iter(group_by).then(
                lambda cols: exp.Group(expressions=cols.collect(list))
            )
            return group.unwrap() if group.is_some() else None

        pivot = exp.Pivot(
            this=exp.to_table("rel"),  # pyright: ignore[reportUnknownMemberType]
            expressions=try_iter(on).collect().into(_on_exprs),
            using=try_iter(using),
            group=_group(),
        )
        qry = (
            try_iter(order_by)
            .collect()
            .then(
                lambda cols: (
                    exp.select("*")  # pyright: ignore[reportUnknownMemberType]
                    .from_(exp.Subquery(this=pivot))
                    .order_by(*cols)
                    .sql(dialect="duckdb")
                )
            )
            .unwrap_or_else(lambda: pivot.sql(dialect="duckdb"))
        )

        return self.__class__(from_query(qry, rel=self.inner()))

    def unpivot(
        self,
        on: TryIter[str],
        index: TryIter[str],
        variable_name: str,
        value_name: str,
        order_by: TryIter[str] = None,
    ) -> Self:
        """Unpivot from wide to long format."""
        index_cols = try_iter(index).collect()
        on_exprs = (
            try_iter(on)
            .then_some()
            .unwrap_or_else(
                lambda: self.columns.iter().filter(lambda name: name not in index_cols)
            )
        )
        unpivot = exp.Pivot(
            this=exp.to_table("rel"),  # pyright: ignore[reportUnknownMemberType]
            expressions=on_exprs,
            unpivot=True,
            into=exp.UnpivotColumns(this=variable_name, expressions=(value_name,)),
        )
        query = (
            index_cols.iter()
            .chain((variable_name, value_name))
            .into(lambda x: exp.select(*x))  # pyright: ignore[reportUnknownMemberType]
            .from_(exp.Subquery(this=unpivot))
        )
        qry = (
            try_iter(order_by)
            .then(lambda cols: query.order_by(*cols))  # pyright: ignore[reportUnknownMemberType]
            .unwrap_or(query)
            .sql(dialect="duckdb")
        )

        return self.__class__(from_query(qry, rel=self.inner()))
