from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from enum import IntEnum, auto
from typing import TYPE_CHECKING, NamedTuple, Self, override

import pyochain as pc
from duckdb import ColumnExpression, SQLExpression

from . import sql
from ._schema import Schema
from .sql.utils import TryIter, try_chain, try_iter

if TYPE_CHECKING:
    from pyochain.traits import PyoCollection, PyoIterable

    from ._datatypes import DataType
    from .sql.typing import IntoExpr, IntoExprColumn
EMPTY_MARKER = "__pql_empty__"
SENTINEL = "__pql_selector__"
TEMP_NAME = "__pql_temp__"
SENTINEL_COL = sql.col(SENTINEL)

type ColumnResolver = Callable[[Schema], PyoCollection[str]]


class ExprKind(IntEnum):
    ROW = auto()
    SCALAR = auto()
    WINDOW = auto()
    UNIQUE = auto()


@dataclass(slots=True)
class ExprMeta(ABC):
    """Metadata for expressions, used for tracking properties that affect query generation."""

    root_name: str
    alias_name: pc.Option[Callable[[str], str]] = field(default_factory=lambda: pc.NONE)
    kind: ExprKind = ExprKind.ROW

    @abstractmethod
    def resolve(
        self, template: sql.SqlExpr, schema: Schema, alias_override: pc.Option[str]
    ) -> pc.Iter[ResolvedExpr]: ...

    def resolve_output_names(
        self, base_names: PyoCollection[str], forced_name: pc.Option[str]
    ) -> PyoCollection[str]:
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
class SingleMeta(ExprMeta):
    @override
    def resolve(
        self, template: sql.SqlExpr, schema: Schema, alias_override: pc.Option[str]
    ) -> pc.Iter[ResolvedExpr]:
        output_names = self.resolve_output_names(
            pc.Seq((self.root_name,)), alias_override
        )
        return ResolvedExpr(template, output_names.first(), self.kind).into_iter()


@dataclass(slots=True)
class MultiMeta(ExprMeta):
    resolver: ColumnResolver = field(kw_only=True)

    @override
    def resolve(
        self, template: sql.SqlExpr, schema: Schema, alias_override: pc.Option[str]
    ) -> pc.Iter[ResolvedExpr]:
        base_names = self.resolver(schema)
        output_names = self.resolve_output_names(base_names, alias_override)
        match alias_override.is_none():
            case True:
                return (
                    base_names.iter()
                    .zip(output_names)
                    .map_star(
                        lambda column_name, output_name: ResolvedExpr(
                            _replace_col(template, column_name), output_name, self.kind
                        )
                    )
                )
            case False:
                return ResolvedExpr(
                    template, output_names.first(), self.kind
                ).into_iter()


class ResolvedExpr(NamedTuple):
    """A fully resolved expression ready for SQL emission."""

    expr: sql.SqlExpr
    name: str
    kind: ExprKind

    def implode_or_scalar(self) -> sql.SqlExpr:
        match self.kind:
            case ExprKind.SCALAR:
                return self.expr.alias(self.name)
            case ExprKind.UNIQUE:
                return self.expr.implode().list.distinct().alias(self.name)
            case _:
                return self.expr.implode().alias(self.name)

    def as_aliased(self) -> sql.SqlExpr:
        return self.expr.alias(self.name)

    def into_iter(self) -> pc.Iter[Self]:
        return pc.Iter.once(self)


@dataclass(slots=True, init=False)
class ExprPlan:
    schema: Schema
    projections: pc.Seq[ResolvedExpr]

    def __init__(
        self,
        schema: Schema,
        exprs: TryIter[IntoExpr],
        more_exprs: Iterable[IntoExpr],
        named_exprs: dict[str, IntoExpr],
    ) -> None:

        def _resolve(
            val: IntoExpr, alias_override: pc.Option[str] = pc.NONE
        ) -> pc.Iter[ResolvedExpr]:
            from ._expr import Expr

            match val:
                case Expr() as expr:
                    return expr.meta.resolve(expr.inner(), schema, alias_override)
                case _:
                    resolved = sql.into_expr(val, as_col=True)
                    output_name = alias_override.unwrap_or(resolved.inner().get_name())
                    return ResolvedExpr(
                        resolved, output_name, kind=ExprKind.ROW
                    ).into_iter()

        self.schema = schema
        expr_map = (
            pc.Iter(named_exprs.items())
            .map_star(lambda k, v: _resolve(v, pc.Some(k)))
            .flatten()
            .collect()
        )
        self.projections = (
            try_chain(exprs, more_exprs).flat_map(_resolve).chain(expr_map).collect()
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
                    return self.aliased_sql().into(
                        lambda exprs: lf.select(*exprs).distinct()
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

    def with_columns_context(self, lf: sql.SqlFrame) -> sql.SqlFrame:
        def _resolve(lf: sql.SqlFrame) -> sql.SqlFrame:
            match self.projections.any(lambda r: r.kind == ExprKind.SCALAR):
                case True:
                    return self.resolve().into(lf.aggregate)
                case False:
                    return self.resolve().into(lambda exprs: lf.select(*exprs))

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

    def resolve(self) -> pc.Iter[sql.SqlExpr]:

        def _resolved(updates: pc.Dict[str, sql.SqlExpr]) -> pc.Iter[sql.SqlExpr]:
            match updates.any(lambda name: name in self.schema):
                case False:
                    return (
                        updates.items()
                        .iter()
                        .map_star(lambda name, e: e.alias(name))
                        .insert(sql.all())
                    )
                case True:
                    return (
                        self.schema.iter()
                        .map(
                            lambda name: updates.get_item(name).map_or(
                                sql.col(name), lambda c: c.alias(name)
                            )
                        )
                        .chain(
                            updates.items()
                            .iter()
                            .filter_star(lambda name, _expr: name not in self.schema)
                            .map_star(lambda name, e: e.alias(name))
                        )
                    )

        return (
            self.projections.iter()
            .map(lambda r: (r.name, r.expr))
            .collect(pc.Dict)
            .into(_resolved)
        )


def _replace_col(template: sql.SqlExpr, column_name: str) -> sql.SqlExpr:
    raw_str = template.pipe(str).replace(SENTINEL, str(ColumnExpression(column_name)))
    return sql.SqlExpr(SQLExpression(raw_str))


def all_columns_resolver() -> ColumnResolver:
    return Schema.keys


def all_fn_resolver(exclude: pc.Option[TryIter[IntoExprColumn]]) -> ColumnResolver:
    return exclude.map(
        lambda exc: (
            try_iter(exc)
            .map(lambda value: sql.into_expr(value, as_col=True).get_name())
            .collect(pc.Set)
            .into(exclude_resolver)
        )
    ).unwrap_or(all_columns_resolver())


def exclude_resolver(excluded: pc.Set[str]) -> ColumnResolver:
    return lambda schema: schema.iter().filter(lambda n: n not in excluded).collect()


def agg_expr_resolver(cols: pc.Option[pc.Seq[str]]) -> ColumnResolver:
    return cols.map(fixed_resolver).unwrap_or(all_columns_resolver())


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
    return lambda schema: schema.iter().filter(predicate).collect()


def difference_resolver(left: ColumnResolver, right: ColumnResolver) -> ColumnResolver:
    def _fn(schema: Schema) -> PyoCollection[str]:
        right_resolver = right(schema)
        return left(schema).iter().filter(lambda n: n not in right_resolver).collect()

    return _fn


def complement_resolver(resolver: ColumnResolver) -> ColumnResolver:
    def _fn(schema: Schema) -> pc.Seq[str]:
        excluded = resolver(schema)
        return schema.iter().filter(lambda n: n not in excluded).collect()

    return _fn


def intersection_resolver(
    left: ColumnResolver, right: ColumnResolver
) -> ColumnResolver:
    def _fn(schema: Schema) -> pc.Seq[str]:
        right_set = right(schema)
        return left(schema).iter().filter(lambda n: n in right_set).collect()

    return _fn


def union_resolver(left: ColumnResolver, right: ColumnResolver) -> ColumnResolver:
    def _fn(schema: Schema) -> pc.Seq[str]:
        selected = left(schema).iter().chain(right(schema)).collect(pc.Set)
        return schema.iter().filter(lambda n: n in selected).collect()

    return _fn
