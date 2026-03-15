from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from enum import IntEnum, auto
from typing import TYPE_CHECKING, NamedTuple, Self

import pyochain as pc
from duckdb import ColumnExpression, SQLExpression

from . import sql
from .sql.utils import TryIter, try_iter

if TYPE_CHECKING:
    from pyochain.traits import PyoCollection, PyoIterable, PyoKeysView

    from ._datatypes import DataType
    from ._schema import Schema
    from .sql.typing import IntoExpr, IntoExprColumn
EMPTY_MARKER = "__pql_empty__"
SENTINEL = "__pql_selector__"
SELECTOR = "__selector__"
TEMP_NAME = "__pql_temp__"
SENTINEL_COL = sql.col(SENTINEL)

type ColumnResolver = Callable[[Schema], PyoCollection[str]]


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
        return cls(SELECTOR, resolver=pc.Some(resolver))

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


@dataclass(slots=True, init=False)
class ExprPlan:
    projections: pc.Seq[ResolvedExpr]

    def __init__(
        self, schema: Schema, exprs: pc.Iter[IntoExpr], named_exprs: dict[str, IntoExpr]
    ) -> None:
        expr_map = (
            pc.Iter(named_exprs.items())
            .map_star(lambda k, v: _projection_resolver(schema, v, pc.Some(k)))
            .flatten()
            .collect()
        )
        self.projections = (
            exprs.flat_map(lambda value: _projection_resolver(schema, value))
            .chain(expr_map)
            .collect()
        )

    def aliased_sql(self) -> pc.Iter[sql.SqlExpr]:
        return self.projections.iter().map(ResolvedExpr.as_aliased)

    def windowed(self, lf: sql.SqlFrame) -> sql.SqlFrame:
        match self.projections.any(lambda p: TEMP_NAME in str(p.expr)):
            case True:
                return lf.select(
                    sql.row_number().over().sub(1).alias(TEMP_NAME), sql.all()
                )
            case False:
                return lf

    def select_context(self, lf: sql.SqlFrame) -> sql.SqlFrame:
        def _non_empty_slct(
            projs: pc.Seq[ResolvedExpr], lf: sql.SqlFrame
        ) -> sql.SqlFrame:
            match projs.all(lambda r: r.kind == ExprKind.UNIQUE):
                case True:
                    return lf.unique(
                        projs.iter().map(lambda r: r.as_unique()).join(", ")
                    )
                case False:
                    match projs.all(lambda r: r.kind == ExprKind.SCALAR):
                        case True:
                            return self.aliased_sql().into(lf.aggregate)
                        case False:
                            return self.aliased_sql().into(
                                lambda exprs: lf.select(*exprs)
                            )

        return self.projections.then(
            lambda projs: _non_empty_slct(projs, self.windowed(lf))
        ).unwrap_or_else(lambda: sql.SqlFrame({EMPTY_MARKER: ()}))

    def with_columns_context(
        self, lf: sql.SqlFrame, col_keys: PyoCollection[str]
    ) -> sql.SqlFrame:
        def _resolve(lf: sql.SqlFrame) -> sql.SqlFrame:
            match self.projections.any(lambda r: r.kind == ExprKind.SCALAR):
                case True:
                    return self.resolve(col_keys).into(lf.aggregate)
                case False:
                    return self.resolve(col_keys).into(lambda exprs: lf.select(*exprs))

        return self.windowed(lf).pipe(_resolve)

    def with_fields_context(self, expr: sql.SqlExpr) -> sql.SqlExpr:
        return self.aliased_sql().into(lambda args: expr.struct.insert(*args))

    def group_by_all_context(self, lf: sql.SqlFrame) -> sql.SqlFrame:
        return self.aliased_sql().into(lf.aggregate, "ALL")

    def agg_context(
        self,
        keys: PyoIterable[sql.SqlExpr],
        aggregator: Callable[[pc.Iter[sql.SqlExpr]], sql.SqlFrame],
    ) -> sql.SqlFrame:
        plan = self.projections.iter().map(lambda p: p.implode_or_scalar())

        return keys.iter().chain(plan).into(aggregator)

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
            self.projections.iter()
            .map(lambda r: (r.name, r.expr))
            .collect(pc.Dict)
            .into(_resolved)
        )


def _projection_resolver(
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
    raw_str = template.pipe(str).replace(SENTINEL, str(ColumnExpression(column_name)))
    return sql.SqlExpr(SQLExpression(raw_str))


def all_columns_resolver(schema: Schema) -> PyoKeysView[str]:
    return schema.keys()


def all_fn_resolver(exclude: pc.Option[Iterable[IntoExprColumn]]) -> ColumnResolver:
    return exclude.map(
        lambda exc: (
            pc.Iter(exc)
            .map(lambda value: sql.into_expr(value, as_col=True).get_name())
            .collect(pc.Set)
            .into(exclude_resolver)
        )
    ).unwrap_or(all_columns_resolver)


def exclude_resolver(excluded: pc.Set[str]) -> ColumnResolver:
    return lambda schema: (
        schema.keys().iter().filter(lambda n: n not in excluded).collect()
    )


def agg_expr_resolver(cols: pc.Option[pc.Seq[str]]) -> ColumnResolver:
    return cols.map(fixed_resolver).unwrap_or(all_columns_resolver)


def fixed_resolver(names: pc.Seq[str]) -> ColumnResolver:
    return lambda _schema: names


def ordered_name_resolver(names: pc.Seq[str]) -> ColumnResolver:
    return lambda schema: names.iter().filter(lambda name: name in schema).collect()


def dtype_resolver(*on: type[DataType]) -> ColumnResolver:
    return lambda schema: (
        schema.items()
        .iter()
        .filter_star(lambda _, dtype: isinstance(dtype, on))
        .map_star(lambda name, _: name)
        .collect()
    )


def name_resolver(predicate: Callable[[str], bool]) -> ColumnResolver:
    return lambda schema: schema.keys().iter().filter(predicate).collect()


def difference_resolver(
    schema: Schema, left: ColumnResolver, right: ColumnResolver
) -> pc.Seq[str]:
    right_resolver = right(schema)
    return left(schema).iter().filter(lambda n: n not in right_resolver).collect()


def complement_resolver(schema: Schema, resolver: ColumnResolver) -> pc.Seq[str]:
    excluded = resolver(schema)
    return schema.keys().iter().filter(lambda n: n not in excluded).collect()


def intersection_resolver(
    schema: Schema, left: ColumnResolver, right: ColumnResolver
) -> pc.Seq[str]:
    right_set = right(schema)
    return left(schema).iter().filter(lambda n: n in right_set).collect()


def union_resolver(
    schema: Schema, left: ColumnResolver, right: ColumnResolver
) -> pc.Seq[str]:
    selected = left(schema).iter().chain(right(schema)).collect(pc.Set)
    return schema.keys().iter().filter(lambda n: n in selected).collect()
