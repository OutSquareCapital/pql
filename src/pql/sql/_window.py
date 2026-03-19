from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import auto
from functools import partial
from typing import TYPE_CHECKING, Literal, NamedTuple, Self, TypedDict, Unpack

import pyochain as pc
from duckdb import Expression, SQLExpression
from sqlglot import exp, parse_one  # pyright: ignore[reportUnknownVariableType]

from .utils import TryIter, UpperStrEnum, try_iter

if TYPE_CHECKING:
    from .typing import FrameMode, IntoExprColumn, WindowExclude


type FrameBound = int | Bounds | str

parse_duckdb = partial(parse_one, dialect="duckdb")


class NullsClause(UpperStrEnum):
    LAST = "NULLS LAST"
    FIRST = "NULLS FIRST"

    @classmethod
    def order(cls, *, last: bool) -> Literal[NullsClause.LAST, NullsClause.FIRST]:
        return cls.LAST if last else cls.FIRST


class SortClause(UpperStrEnum):
    ASC = auto()
    DESC = auto()

    @classmethod
    def order(cls, *, desc: bool) -> Literal[SortClause.ASC, SortClause.DESC]:
        return cls.DESC if desc else cls.ASC


class Bounds(UpperStrEnum):
    PRECEDING = auto()
    FOLLOWING = auto()
    UNBOUNDED = auto()
    CURRENT = auto()
    ROW = auto()


class DirectionArgs(TypedDict):
    descending: TryIter[bool]
    nulls_last: TryIter[bool]


class FnArgs(TypedDict):
    fn_order_by: pc.Option[TryIter[IntoExprColumn]]
    fn_descending: TryIter[bool]
    fn_nulls_last: TryIter[bool]


class ClauseArgs(TypedDict):
    partition_by: pc.Option[pc.Seq[exp.Expr]]
    order: pc.Option[exp.Order]
    spec: pc.Option[exp.WindowSpec]


class BoundArgs(TypedDict):
    frame_start: pc.Option[FrameBound]
    frame_end: pc.Option[FrameBound]
    exclude: pc.Option[WindowExclude]


def get_order(
    order_by: pc.Option[TryIter[IntoExprColumn]], **kwargs: Unpack[DirectionArgs]
) -> pc.Option[exp.Order]:
    return order_by.map(lambda x: try_iter(x).collect()).map(
        lambda cols: exp.Order(expressions=_ordered(cols, **kwargs))
    )


def get_partition(
    partition_by: pc.Option[TryIter[IntoExprColumn]],
) -> pc.Option[pc.Seq[exp.Expr]]:
    return partition_by.map(try_iter).map(
        lambda cols: cols.map(lambda c: parse_duckdb(str(c))).collect()
    )


def _ordered(
    cols: pc.Seq[IntoExprColumn], **kwargs: Unpack[DirectionArgs]
) -> pc.Seq[exp.Ordered]:
    def _expand_clauses(*, clauses: TryIter[bool], n: int) -> pc.Iter[bool]:
        match clauses:
            case Iterable() as seq:
                return pc.Iter(seq)
            case _ as val:
                return try_iter(val).cycle().take(n)

    return (
        cols.iter()
        .zip(
            _expand_clauses(clauses=kwargs["descending"], n=cols.length()),
            _expand_clauses(clauses=kwargs["nulls_last"], n=cols.length()),
        )
        .map_star(
            lambda item, desc, nl: exp.Ordered(
                this=parse_duckdb(str(item)), desc=desc, nulls_first=not nl
            )
        )
        .collect()
    )


@dataclass(slots=True)
class OverBuilder:
    expr: exp.Expr

    @classmethod
    def from_expr(cls, expr: Expression) -> Self:
        return cls(parse_duckdb(expr.get_name()))

    def handle_nulls(self, *, ignore_nulls: bool) -> Self:
        match ignore_nulls:
            case True:
                return self.__class__(exp.IgnoreNulls(this=self.expr))
            case False:
                return self

    def handle_distinct(self, *, distinct: bool) -> Self:
        match distinct:
            case True:
                copied = self.expr.copy()
                match copied.find(exp.Func):
                    case exp.Func() as fn:
                        arg: exp.Expr | None = fn.this  # pyright: ignore[reportAny]
                        match arg:
                            case exp.Expr() as a:
                                fn.set("this", exp.Distinct(expressions=[a]))
                            case _:
                                pass
                    case _:
                        pass
                return self.__class__(copied)
            case False:
                return self

    def handle_fn_order_by(self, **kwargs: Unpack[FnArgs]) -> Self:
        def _build(cols: pc.Seq[IntoExprColumn]) -> exp.WithinGroup:
            exprs = _ordered(
                cols,
                descending=kwargs["fn_descending"],
                nulls_last=kwargs["fn_nulls_last"],
            )
            return exp.WithinGroup(
                this=self.expr,
                expression=exp.Order(expressions=exp.Order(expressions=exprs)),
            )

        return (
            kwargs["fn_order_by"]
            .map(lambda x: try_iter(x).collect().into(_build))
            .map(self.__class__)
            .unwrap_or(self)
        )

    def handle_filter(self, filter_cond: pc.Option[IntoExprColumn]) -> Self:
        return (
            filter_cond.map(
                lambda c: exp.Filter(
                    this=self.expr,
                    expression=exp.Where(
                        this=parse_duckdb(str(c)),
                    ),
                )
            )
            .map(self.__class__)
            .unwrap_or(self)
        )

    def handle_clauses(self, **kwargs: Unpack[ClauseArgs]) -> Self:
        match self.expr.find(exp.Window):
            case None:
                return self.__class__(_wrap_in_window(self.expr, **kwargs))
            case _:
                return self.__class__(_inject_into_existing(self.expr, **kwargs))

    def build(self) -> Expression:
        return SQLExpression(self.expr.sql(dialect="duckdb"))

    def build_fn(
        self, *, ignore_nulls: bool = False, **kwargs: Unpack[FnArgs]
    ) -> Expression:
        return (
            self.handle_fn_order_by(**kwargs)
            .handle_nulls(ignore_nulls=ignore_nulls)
            .build()
        )


def _inject_into_existing(expr: exp.Expr, **kwargs: Unpack[ClauseArgs]) -> exp.Expr:
    inj = partial(_inject, with_spec=False, **kwargs)
    pc.Iter(expr.find_all(exp.Window)).for_each(inj)
    return expr


def _wrap_in_window(expr: exp.Expr, **kwargs: Unpack[ClauseArgs]) -> exp.Window:
    window = exp.Window(this=expr)
    _inject(window, with_spec=True, **kwargs)
    return window


def _inject(w: exp.Window, *, with_spec: bool, **kwargs: Unpack[ClauseArgs]) -> None:
    _ = kwargs["partition_by"].map(lambda pb: w.set("partition_by", pb))
    _ = kwargs["order"].map(lambda o: w.set("order", o))
    if with_spec:
        _ = kwargs["spec"].map(lambda s: w.set("spec", s))


def make_spec(
    mode: FrameMode, *, has_order_by: bool, **kwargs: Unpack[BoundArgs]
) -> pc.Option[exp.WindowSpec]:
    return (
        BoundsValues.new(has_order_by=has_order_by, **kwargs)
        .map(lambda b: b.into_spec(mode))
        .inspect(
            lambda spec: kwargs["exclude"].inspect(lambda ex: spec.set("exclude", ex))
        )
    )


class Side(NamedTuple):
    value: FrameBound
    direction: FrameBound

    @classmethod
    def new(cls, value: FrameBound, direction: Bounds) -> Self:
        """Convert a frame bound value to ``(value, side)`` for `exp.WindowSpec`."""
        match value:
            case 0:
                return cls(Bounds.CURRENT, Bounds.ROW)
            case int(n) if n < 0:
                return cls(str(-n), Bounds.PRECEDING)
            case int(n):
                return cls(str(n), Bounds.FOLLOWING)
            case _:
                return cls(value, direction)


class BoundsValues(NamedTuple):
    start: Side
    end: Side

    @classmethod
    def new(cls, *, has_order_by: bool, **kwargs: Unpack[BoundArgs]) -> pc.Option[Self]:
        match (kwargs["frame_start"], kwargs["frame_end"]):
            case (pc.Some(s), pc.Some(e)):
                return pc.Some(
                    cls(Side.new(s, Bounds.PRECEDING), Side.new(e, Bounds.FOLLOWING))
                )
            case (pc.Some(s), pc.NONE):
                return pc.Some(
                    cls(
                        Side.new(s, Bounds.PRECEDING),
                        Side.new(Bounds.UNBOUNDED, Bounds.FOLLOWING),
                    )
                )
            case (pc.NONE, pc.Some(e)):
                return pc.Some(
                    cls(
                        Side.new(Bounds.UNBOUNDED, Bounds.PRECEDING),
                        Side.new(e, Bounds.FOLLOWING),
                    )
                )
            case _ if has_order_by or kwargs["exclude"].is_some():
                return pc.Some(
                    cls(
                        Side.new(Bounds.UNBOUNDED, Bounds.PRECEDING),
                        Side.new(Bounds.UNBOUNDED, Bounds.FOLLOWING),
                    )
                )
            case _:
                return pc.NONE

    def into_spec(self, mode: FrameMode) -> exp.WindowSpec:
        return exp.WindowSpec(
            kind=mode,
            start=self.start.value,
            start_side=self.start.direction,
            end=self.end.value,
            end_side=self.end.direction,
        )
