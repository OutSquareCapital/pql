from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, NamedTuple, Self, overload, override

import pyochain as pc
from duckdb import SQLExpression
from pyochain.traits import PyoCollection, PyoSequence

from . import sql
from .sql.utils import TryIter, try_iter

if TYPE_CHECKING:
    from ._schema import ColumnResolver, Schema
    from .sql.typing import IntoExpr

SENTINEL = "__pql_selector__"
SENTINEL_COL = sql.col(SENTINEL)


class ExprKind(IntEnum):
    ROW = auto()
    SCALAR = auto()
    WINDOW = auto()
    UNIQUE = auto()


@dataclass(slots=True)
class ExprMeta:
    """Metadata for expressions, used for tracking properties that affect query generation."""

    root_name: str
    alias_name: pc.Option[Callable[[str], str]] = field(default_factory=lambda: pc.NONE)
    kind: ExprKind = ExprKind.ROW
    resolver: pc.Option[ColumnResolver] = field(default_factory=lambda: pc.NONE)

    @classmethod
    def from_selector(cls, resolver: ColumnResolver) -> Self:
        return cls("__selector__", resolver=pc.Some(resolver))

    @classmethod
    def from_horizontal(cls, exprs: TryIter[IntoExpr]) -> Self:
        return (
            try_iter(exprs)
            .next()
            .map(lambda v: sql.into_expr(v, as_col=True).get_name())
            .map(cls)
            .unwrap()
        )

    @classmethod
    def from_agg_expr(
        cls, cols: pc.Option[pc.Seq[str]], resolver: ColumnResolver
    ) -> Self:
        return cls(
            cols.map(lambda c: c.first()).unwrap_or("all"),
            kind=ExprKind.SCALAR,
            resolver=pc.Some(resolver),
        )

    @classmethod
    def from_all(cls, resolver: ColumnResolver) -> Self:
        return cls("all", resolver=pc.Some(resolver))

    @property
    def is_multi(self) -> bool:
        return self.resolver.is_some()

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


class ResolvedExpr(NamedTuple):
    """A fully resolved expression ready for SQL emission."""

    expr: sql.SqlExpr
    name: str
    kind: ExprKind

    def implode_or_scalar(self) -> sql.SqlExpr:
        return (
            self.expr.alias(self.name)
            if self.kind == ExprKind.SCALAR
            else self.expr.implode().alias(self.name)
        )

    def as_aliased(self) -> sql.SqlExpr:
        return self.expr.alias(self.name)

    def as_unique(self) -> str:
        base_sql = str(self.expr)
        return base_sql if self.name == base_sql else f"{base_sql} AS {self.name}"


@dataclass(slots=True)
class ExprPlan(PyoSequence[ResolvedExpr]):
    projections: pc.Seq[ResolvedExpr]

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
                    pc.Iter(mapping.items())
                    .map_star(lambda k, v: _resolve_projection(schema, v, pc.Some(k)))
                    .flatten()
                )
            )
            .unwrap_or_else(pc.Iter[ResolvedExpr].new)
        )
        return cls(
            exprs.flat_map(lambda value: _resolve_projection(schema, value))
            .chain(expr_map)
            .collect()
        )

    @override
    def __iter__(self) -> Iterator[ResolvedExpr]:
        return iter(self.projections)

    @override
    def __len__(self) -> int:
        return len(self.projections)

    @overload
    def __getitem__(self, index: int) -> ResolvedExpr: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[ResolvedExpr]: ...
    @override
    def __getitem__(
        self,
        index: int | slice[Any, Any, Any],  # pyright: ignore[reportExplicitAny] #pragma: no cover
    ) -> ResolvedExpr | Sequence[ResolvedExpr]:
        return self.projections[index]

    def aliased_sql(self) -> pc.Iter[sql.SqlExpr]:
        return self.iter().map(ResolvedExpr.as_aliased)

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

        return (
            self.iter().map(lambda r: (r.name, r.expr)).collect(pc.Dict).into(_resolved)
        )


def _resolve_projection(
    schema: Schema, value: IntoExpr, alias_override: pc.Option[str] = pc.NONE
) -> pc.Iter[ResolvedExpr]:
    from ._expr import Expr

    into_resolved = pc.Iter[ResolvedExpr].once
    match value:
        case Expr() as expr:
            base_names = expr.meta.resolver.map(lambda r: r(schema)).unwrap_or_else(
                lambda: pc.Seq((expr.meta.root_name,))
            )
            output_names = expr.meta.resolve_output_names(base_names, alias_override)
            kind = expr.meta.kind
            match expr.meta.is_multi and alias_override.is_none():
                case True:
                    template = expr.inner()
                    return (
                        base_names.iter()
                        .zip(output_names)
                        .map_star(
                            lambda column_name, output_name: ResolvedExpr(
                                _replace_col(template, column_name), output_name, kind
                            )
                        )
                    )
                case False:
                    return into_resolved(
                        ResolvedExpr(expr.inner(), output_names.first(), kind)
                    )
        case _:
            resolved = sql.into_expr(value, as_col=True)
            output_name = alias_override.unwrap_or(resolved.inner().get_name())
            return into_resolved(ResolvedExpr(resolved, output_name, kind=ExprKind.ROW))


def _replace_col(template: sql.SqlExpr, column_name: str) -> sql.SqlExpr:
    return sql.SqlExpr(SQLExpression(template.pipe(str).replace(SENTINEL, column_name)))
