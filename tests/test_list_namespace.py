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
                        [1, 2, 7, 3],
                        [3, 4, 5, 5],
                        [2, 5],
                        [1],
                    ],
                    "x_var": [
                        [],
                        [1, 2],
                        [5],
                        [1],
                    ],
                    "y": [1, 2, 5, 0],
                    "z": [7, 6, 5, 0],
                    "str_vals": [
                        [
                            "g",
                            "b",
                            "c",
                        ],
                        ["a", "b"],
                        ["c"],
                        [],
                    ],
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
    assert_eq(pql.col("x").list.len(), nw.col("x").list.len())
    assert_eq_pl(pql.col("x_var").list.len(), pl.col("x_var").list.len())


def test_list_unique() -> None:
    assert_eq(pql.col("x").list.unique(), nw.col("x").list.unique())


def test_list_contains() -> None:
    assert_eq(pql.col("x").list.contains(2), nw.col("x").list.contains(2))
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
        pql.col("x").list.get(0),
        nw.col("x").list.get(0),
    )
    assert_eq_pl(
        pql.col("x").list.get(-1),
        pl.col("x").list.get(-1),
    )
    with pytest.raises(pl.exceptions.ComputeError, match="get index is out of bounds"):
        assert_eq_pl(
            pql.col("x_var").list.get(10),
            pl.col("x_var").list.get(10),
        )


def test_list_min() -> None:
    assert_eq(pql.col("x").list.min(), nw.col("x").list.min())


def test_list_max() -> None:
    assert_eq(pql.col("x").list.max(), nw.col("x").list.max())


def test_list_mean() -> None:
    assert_eq(pql.col("x").list.mean(), nw.col("x").list.mean())


def test_list_median() -> None:
    assert_eq(pql.col("x").list.median(), nw.col("x").list.median())


def test_list_sum() -> None:
    assert_eq(pql.col("x").list.sum(), nw.col("x").list.sum())


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


def test_list_eval_num() -> None:
    assert_eq_pl(
        pql.col("x").list.eval(pql.element().mul(2)),
        pl.col("x").list.eval(pl.element().mul(2)),
    )


def test_list_eval_str() -> None:
    assert_eq_pl(
        pql.col("str_vals").list.eval(pql.element().str.to_uppercase()),
        pl.col("str_vals").list.eval(pl.element().str.to_uppercase()),
    )
