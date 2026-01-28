from collections.abc import Iterable

import duckdb
import narwhals as nw
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pql


def sample_df() -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    """Create a sample DataFrame with list data for testing."""
    return nw.from_native(
        duckdb.from_arrow(
            pl.DataFrame(
                {
                    "x": [
                        [1, 2, None, 3],
                        [3, 4, None, 5],
                        [None, 5],
                        [None],
                    ],
                    "x_var": [
                        [],
                        [1, None],
                        None,
                        [None],
                    ],
                    "y": [1, 2, 5, 0],
                    "z": [None, None, 5, None],
                },
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


def test_list_len() -> None:
    assert_eq(
        pql.col("x").list.len().alias("x_len"),
        nw.col("x").list.len().alias("x_len"),
    )
    assert_eq_pl(
        pql.col("x_var").list.len().alias("x_len"),
        pl.col("x_var").list.len().alias("x_len"),
    )


def test_list_unique() -> None:
    assert_eq(
        pql.col("x").list.unique().alias("x_unique"),
        nw.col("x").list.unique().alias("x_unique"),
    )


def test_list_contains() -> None:
    assert_eq(
        pql.col("x").list.contains(2).alias("x"),
        nw.col("x").list.contains(2).alias("x"),
    )
    assert_eq_pl(
        (
            pql.col("x").list.contains(None, nulls_equal=True).alias("x_nulls_eq"),
            pql.col("x").list.contains(None, nulls_equal=False).alias("x_nulls_neq"),
            pql.col("x")
            .list.contains(pql.col("y"), nulls_equal=True)
            .alias("x_nulls_eq_y"),
            pql.col("x")
            .list.contains(pql.col("y"), nulls_equal=False)
            .alias("x_nulls_neq_y"),
            pql.col("x").list.contains(pql.col("y")).alias("x_y"),
        ),
        (
            pl.col("x").list.contains(None, nulls_equal=True).alias("x_nulls_eq"),
            pl.col("x").list.contains(None, nulls_equal=False).alias("x_nulls_neq"),
            pl.col("x")
            .list.contains(pl.col("y"), nulls_equal=True)
            .alias("x_nulls_eq_y"),
            pl.col("x")
            .list.contains(pl.col("y"), nulls_equal=False)
            .alias("x_nulls_neq_y"),
            pl.col("x").list.contains(pl.col("y")).alias("x_y"),
        ),
    )


def test_list_get() -> None:
    assert_eq(
        pql.col("x").list.get(0).alias("x"),
        nw.col("x").list.get(0).alias("x"),
    )
    assert_eq_pl(
        pql.col("x").list.get(-1).alias("x"),
        pl.col("x").list.get(-1).alias("x"),
    )
    with pytest.raises(pl.exceptions.ComputeError, match="get index is out of bounds"):
        assert_eq_pl(
            pql.col("x_var").list.get(10).alias("x"),
            pl.col("x_var").list.get(10).alias("x"),
        )


def test_list_min() -> None:
    assert_eq(
        pql.col("x").list.min().alias("x_min"),
        nw.col("x").list.min().alias("x_min"),
    )


def test_list_max() -> None:
    assert_eq(
        pql.col("x").list.max().alias("x_max"),
        nw.col("x").list.max().alias("x_max"),
    )


def test_list_mean() -> None:
    assert_eq(
        pql.col("x").list.mean().alias("x_mean"),
        nw.col("x").list.mean().alias("x_mean"),
    )


def test_list_median() -> None:
    assert_eq(
        pql.col("x").list.median().alias("x_median"),
        nw.col("x").list.median().alias("x_median"),
    )


def test_list_sum() -> None:
    assert_eq(
        pql.col("x").list.sum().alias("x_sum"),
        nw.col("x").list.sum().alias("x_sum"),
    )


def test_list_sort() -> None:
    assert_eq(
        (
            pql.col("x").list.sort().alias("x_sorted"),
            pql.col("x")
            .list.sort(descending=True, nulls_last=True)
            .alias("x_sorted_desc"),
        ),
        (
            nw.col("x").list.sort().alias("x_sorted"),
            nw.col("x")
            .list.sort(descending=True, nulls_last=True)
            .alias("x_sorted_desc"),
        ),
    )
