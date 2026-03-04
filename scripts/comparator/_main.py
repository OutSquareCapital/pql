"""Compare pql API coverage against Polars."""

import narwhals as nw
import polars as pl
import pyochain as pc
from narwhals.group_by import LazyGroupBy as nwLazyGroupBy
from polars.lazyframe.group_by import LazyGroupBy as plLazyGroupBy

import pql
from pql._frame import LazyGroupBy as pqlLazyGroupBy

from .._utils import Pql
from ._rules import IGNORED_MEMBERS
from ._text import ClassComparison, header, render_summary_table


def get_comparisons() -> str:
    pl_col = pl.col("x")
    pql_col = pql.col("x")
    nw_col = pc.Some(nw.col("x"))
    return (
        pc.Iter(
            (
                ClassComparison(
                    pc.Some(nw.LazyFrame),
                    pl.LazyFrame,
                    pql.LazyFrame,
                    Pql.LAZY_FRAME,
                    ignored_names=IGNORED_MEMBERS.get_item(Pql.LAZY_FRAME).unwrap(),
                ),
                ClassComparison(pc.Some(nw.Expr), pl.Expr, pql.Expr, Pql.EXPR),
                ClassComparison(
                    pc.Some(nwLazyGroupBy),
                    plLazyGroupBy,
                    pqlLazyGroupBy,
                    Pql.LAZY_GROUP_BY,
                ),
                ClassComparison(
                    nw_col.map(lambda x: x.str.__class__),
                    pl_col.str.__class__,
                    pql_col.str.__class__,
                    Pql.EXPR_STR_NAME_SPACE,
                ),
                ClassComparison(
                    nw_col.map(lambda x: x.list.__class__),
                    pl_col.list.__class__,
                    pql_col.list.__class__,
                    Pql.EXPR_LIST_NAME_SPACE,
                ),
                ClassComparison(
                    nw_col.map(lambda x: x.struct.__class__),
                    pl_col.struct.__class__,
                    pql_col.struct.__class__,
                    Pql.EXPR_STRUCT_NAME_SPACE,
                ),
                ClassComparison(
                    nw_col.map(lambda x: x.name.__class__),
                    pl_col.name.__class__,
                    pql_col.name.__class__,
                    Pql.EXPR_NAME_NAME_SPACE,
                ),
                ClassComparison(
                    pc.NONE,
                    pl_col.arr.__class__,
                    pql_col.arr.__class__,
                    Pql.EXPR_ARR_NAME_SPACE,
                ),
                ClassComparison(
                    nw_col.map(lambda x: x.dt.__class__),
                    pl_col.dt.__class__,
                    pql_col.dt.__class__,
                    Pql.EXPR_DT_NAME_SPACE,
                ),
                ClassComparison(
                    pc.Some(nw),
                    pl,
                    pql,
                    Pql.MODULE_FUNCTIONS,
                    ignored_names=IGNORED_MEMBERS.get_item(
                        Pql.MODULE_FUNCTIONS
                    ).unwrap(),
                ),
            )
        )
        .map(lambda comp: comp.to_report())
        .collect()
        .into(
            lambda comps: (
                header()
                .chain(render_summary_table(comps))
                .chain(comps.iter().flat_map(lambda comp: comp.to_section()))
            )
        )
        .join("\n")
    )
