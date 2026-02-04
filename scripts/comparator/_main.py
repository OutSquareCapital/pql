"""Compare pql API coverage against Polars."""

import narwhals as nw
import polars as pl
import pyochain as pc

import pql

from ._text import ClassComparison, header, render_summary_table


def get_comparisons() -> str:
    return (
        pc.Iter(
            (
                ClassComparison(nw.LazyFrame, pl.LazyFrame, pql.LazyFrame),
                ClassComparison(nw.Expr, pl.Expr, pql.Expr),
                ClassComparison(
                    nw.col("x").str.__class__,
                    pl.col("x").str.__class__,
                    pql.col("x").str.__class__,
                    name="Expr.str",
                ),
                ClassComparison(
                    nw.col("x").list.__class__,
                    pl.col("x").list.__class__,
                    pql.col("x").list.__class__,
                    name="Expr.list",
                ),
                ClassComparison(
                    nw.col("x").struct.__class__,
                    pl.col("x").struct.__class__,
                    pql.col("x").struct.__class__,
                    name="Expr.struct",
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
