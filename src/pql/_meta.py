from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Self

import pyochain as pc
from pyochain.traits import PyoCollection, PyoIterable

from . import sql

if TYPE_CHECKING:
    from ._schema import ColumnResolver, Schema
    from .sql.typing import IntoExpr


@dataclass(slots=True)
class ExprMeta:
    """Metadata for expressions, used for tracking properties that affect query generation."""

    root_name: str
    alias_name: pc.Option[Callable[[str], str]] = field(default_factory=lambda: pc.NONE)
    is_scalar_like: bool = False
    has_window: bool = False
    is_unique_projection: bool = False
    column_resolver: pc.Option[ColumnResolver] = field(default_factory=lambda: pc.NONE)
    multi_agg: pc.Option[Callable[[sql.SqlExpr], sql.SqlExpr]] = field(
        default_factory=lambda: pc.NONE
    )

    @property
    def is_multi(self) -> bool:
        return self.column_resolver.is_some()

    @property
    def is_scalar_select(self) -> bool:
        return self.is_scalar_like and not self.has_window

    def from_projection(self, output_name: str) -> Self:
        return replace(
            self, root_name=output_name, alias_name=pc.NONE, column_resolver=pc.NONE
        )

    def resolve_output_names(
        self, base_names: pc.traits.PyoCollection[str], forced_name: pc.Option[str]
    ) -> pc.traits.PyoCollection[str]:
        match forced_name:
            case pc.Some(name):
                return pc.Seq((name,))
            case _:
                match self.alias_name:
                    case pc.Some(alias_fn):
                        return base_names.iter().map(alias_fn).collect()
                    case _:
                        return base_names

    def with_alias_mapper(self, mapper: Callable[[str], str]) -> Self:
        match self.alias_name:
            case pc.Some(current):

                def _composed(name: str) -> str:
                    return mapper(current(name))

                composed = _composed
            case _:

                def _composed(name: str) -> str:
                    return mapper(name)

                composed = _composed
        return replace(self, alias_name=pc.Some(composed))

    def clear_alias(self) -> Self:
        return replace(self, alias_name=pc.NONE)


@dataclass(slots=True)
class ExprProjection:
    expr: sql.SqlExpr
    meta: ExprMeta

    def implode_or_scalar(self) -> sql.SqlExpr:
        name = self.meta.root_name
        return (
            self.expr.alias(name)
            if self.meta.is_scalar_select
            else self.expr.implode().alias(name)
        )

    def as_aliased(self) -> sql.SqlExpr:
        return self.expr.alias(self.meta.root_name)

    def as_unique(self) -> str:
        base_sql = str(self.expr)
        return (
            base_sql
            if self.meta.root_name == base_sql
            else f"{base_sql} AS {self.meta.root_name}"
        )


@dataclass(slots=True)
class ExprPlan(PyoIterable[ExprProjection]):
    projections: pc.Seq[ExprProjection]

    @classmethod
    def from_inputs(
        cls,
        schema: Schema,
        exprs: pc.Iter[IntoExpr],
        named_exprs: dict[str, IntoExpr] | None = None,
    ) -> Self:
        expr_map = (
            pc.Option(named_exprs)
            .map(
                lambda mapping: (
                    pc.Dict.from_ref(mapping)
                    .items()
                    .iter()
                    .map_star(
                        lambda k, v: _resolve_projection(
                            schema, v, alias_override=pc.Some(k)
                        )
                    )
                    .flatten()
                )
            )
            .unwrap_or_else(pc.Iter[ExprProjection].new)
        )
        return cls(
            exprs.flat_map(lambda value: _resolve_projection(schema, value))
            .chain(expr_map)
            .collect()
        )

    def __iter__(self) -> Iterator[ExprProjection]:
        return iter(self.projections)

    def aliased_sql(self) -> pc.Iter[sql.SqlExpr]:
        return self.iter().map(lambda p: p.as_aliased())

    def as_unique(self) -> pc.Iter[str]:
        return self.iter().map(lambda p: p.as_unique())

    def to_updates(self) -> pc.Dict[str, sql.SqlExpr]:
        return self.iter().map(lambda p: (p.meta.root_name, p.expr)).collect(pc.Dict)

    def is_scalar_select(self) -> bool:
        return self.all(lambda p: p.meta.is_scalar_select)

    def has_scalar(self) -> bool:
        return self.any(lambda p: p.meta.is_scalar_select)

    def can_use_unique(self) -> bool:
        return self.all(lambda p: p.meta.is_unique_projection)

    def as_result(self) -> pc.Option[Self]:
        """Return `Some(self)` if non-empty, `NONE` otherwise.

        This way, we can avoid runtime crash in case of empty selections/aggregations (e.g on an empty selector result).
        """
        return self.projections.then(lambda _: self)

    def resolve(self, col_keys: PyoCollection[str]) -> pc.Iter[sql.SqlExpr]:

        def _resolved(updates: pc.Dict[str, sql.SqlExpr]) -> pc.Iter[sql.SqlExpr]:
            match updates.keys().any(lambda name: name in col_keys):
                case False:
                    return (
                        updates.items()
                        .iter()
                        .map_star(lambda name, e: e.alias(name))
                        .insert(sql.all())
                    )
                case True:
                    return (
                        col_keys.iter()
                        .map(
                            lambda name: updates.get_item(name).map_or(
                                sql.col(name), lambda c: c.alias(name)
                            )
                        )
                        .chain(
                            updates.items()
                            .iter()
                            .filter_star(lambda name, _expr: name not in col_keys)
                            .map_star(lambda name, e: e.alias(name))
                        )
                    )

        return self.to_updates().into(_resolved)


def _resolve_projection(
    schema: Schema, value: IntoExpr, *, alias_override: pc.Option[str] = pc.NONE
) -> pc.Iter[ExprProjection]:
    from ._expr import Expr

    into_proj = pc.Iter[ExprProjection].once
    match value:
        case Expr() as expr:
            base_names = expr.meta.column_resolver.map(
                lambda resolver: resolver(schema)
            ).unwrap_or_else(lambda: pc.Seq((expr.meta.root_name,)))
            output_names = expr.meta.resolve_output_names(base_names, alias_override)
            match expr.meta.is_multi and alias_override.is_none():
                case True:
                    return (
                        base_names.iter()
                        .zip(output_names)
                        .map_star(
                            lambda column_name, output_name: ExprProjection(
                                expr.meta.multi_agg.map(
                                    lambda agg: sql.col(column_name).pipe(agg)
                                ).unwrap_or(sql.col(column_name)),
                                expr.meta.from_projection(output_name),
                            )
                        )
                    )
                case False:
                    return into_proj(
                        ExprProjection(
                            expr.inner(),
                            expr.meta.from_projection(output_names.first()),
                        )
                    )
        case _:
            resolved = sql.into_expr(value, as_col=True)
            resolved_meta = ExprMeta(resolved.inner().get_name())
            return into_proj(
                ExprProjection(
                    resolved,
                    resolved_meta.from_projection(
                        alias_override.unwrap_or(resolved_meta.root_name)
                    ),
                )
            )
