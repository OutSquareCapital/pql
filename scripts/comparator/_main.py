"""Compare pql API coverage against Polars."""

import narwhals as nw
import polars as pl
import pyochain as pc
from narwhals.group_by import LazyGroupBy as nwLazyGroupBy
from polars.lazyframe.group_by import LazyGroupBy as plLazyGroupBy

import pql
from pql._frame import LazyGroupBy as pqlLazyGroupBy

from .._utils import Pql
from ._text import ClassComparison, header, render_summary_table


def get_comparisons() -> str:
    return (
        pc.Iter(
            (
                ClassComparison(
                    pc.Some(nw.LazyFrame), pl.LazyFrame, pql.LazyFrame, Pql.LAZY_FRAME
                ),
                ClassComparison(pc.Some(nw.Expr), pl.Expr, pql.Expr, Pql.EXPR),
                ClassComparison(
                    pc.Some(nwLazyGroupBy),
                    plLazyGroupBy,
                    pqlLazyGroupBy,
                    Pql.LAZY_GROUP_BY,
                ),
                ClassComparison(
                    pc.Some(nw.col("x").str.__class__),
                    pl.col("x").str.__class__,
                    pql.col("x").str.__class__,
                    Pql.EXPR_STR_NAME_SPACE,
                ),
                ClassComparison(
                    pc.Some(nw.col("x").list.__class__),
                    pl.col("x").list.__class__,
                    pql.col("x").list.__class__,
                    Pql.EXPR_LIST_NAME_SPACE,
                ),
                ClassComparison(
                    pc.Some(nw.col("x").struct.__class__),
                    pl.col("x").struct.__class__,
                    pql.col("x").struct.__class__,
                    Pql.EXPR_STRUCT_NAME_SPACE,
                ),
                ClassComparison(
                    pc.Some(nw.col("x").name.__class__),
                    pl.col("x").name.__class__,
                    pql.col("x").name.__class__,
                    Pql.EXPR_NAME_NAME_SPACE,
                ),
                ClassComparison(
                    pc.NONE,
                    pl.col("x").arr.__class__,
                    pql.col("x").arr.__class__,
                    Pql.EXPR_ARR_NAME_SPACE,
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
