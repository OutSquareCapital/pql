from collections.abc import Iterable

import duckdb
import narwhals as nw
import polars as pl
from polars.testing import assert_frame_equal

import pql


def sample_df() -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    """Create a sample DataFrame with string data for testing."""
    return nw.from_native(
        duckdb.from_arrow(
            pl.DataFrame(
                {
                    "x": [
                        {"a": 1, "b": 2, "c": 3, "d": 4},
                        {"a": 5, "b": 6, "c": 7, "d": 8},
                    ],
                }
            )
        )
    )


def assert_eq(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: nw.Expr | Iterable[nw.Expr]
) -> None:
    assert_frame_equal(
        sample_df().lazy().select(polars_exprs).to_native().pl(),
        pql.LazyFrame(sample_df().to_native()).select(pql_exprs).collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def assert_eq_pl(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: pl.Expr | Iterable[pl.Expr]
) -> None:
    assert_frame_equal(
        sample_df().to_native().pl(lazy=True).select(polars_exprs).collect(),
        pql.LazyFrame(sample_df().to_native()).select(pql_exprs).collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def test_field() -> None:
    assert_eq(
        pql.col("x").struct.field("a").alias("a"),
        nw.col("x").struct.field("a").alias("a"),
    )


def test_with_fields() -> None:
    assert_eq_pl(
        pql.col("x")
        .struct.with_fields(
            pql.col("x").struct.field("a").alias("e"),
            pql.col("x").struct.field("b").alias("f"),
        )
        .alias("x"),
        pl.col("x")
        .struct.with_fields(
            pl.col("x").struct.field("a").alias("e"),
            pl.col("x").struct.field("b").alias("f"),
        )
        .alias("x"),
    )
