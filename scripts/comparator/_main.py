"""Compare pql API coverage against Polars."""

import narwhals as nw
import polars as pl
import pyochain as pc
from narwhals.group_by import LazyGroupBy as nwLazyGroupBy
from polars.lazyframe.group_by import LazyGroupBy as plLazyGroupBy

import pql
from pql._frame import LazyGroupBy as pqlLazyGroupBy

from ._text import ClassComparison, header, render_summary_table


def get_comparisons() -> str:
    return (
        pc.Iter(
            (
                ClassComparison(
                    pc.Some(nw.LazyFrame), pl.LazyFrame, pql.LazyFrame, "LazyFrame"
                ),
                ClassComparison(pc.Some(nw.Expr), pl.Expr, pql.Expr, "Expr"),
                ClassComparison(
                    pc.Some(nwLazyGroupBy), plLazyGroupBy, pqlLazyGroupBy, "LazyGroupBy"
                ),
                ClassComparison(
                    pc.Some(nw.col("x").str.__class__),
                    pl.col("x").str.__class__,
                    pql.col("x").str.__class__,
                    name="Expr.str",
                ),
                ClassComparison(
                    pc.Some(nw.col("x").list.__class__),
                    pl.col("x").list.__class__,
                    pql.col("x").list.__class__,
                    name="Expr.list",
                ),
                ClassComparison(
                    pc.Some(nw.col("x").struct.__class__),
                    pl.col("x").struct.__class__,
                    pql.col("x").struct.__class__,
                    name="Expr.struct",
                ),
                ClassComparison(
                    pc.Some(nw.col("x").name.__class__),
                    pl.col("x").name.__class__,
                    pql.col("x").name.__class__,
                    name="Expr.name",
                ),
                ClassComparison(
                    pc.NONE,
                    pl.col("x").arr.__class__,
                    pql.col("x").arr.__class__,
                    name="Expr.arr",
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
