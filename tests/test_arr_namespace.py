from collections.abc import Iterable

import duckdb
import narwhals as nw
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pql

SCHEMA = {
    "x": pql.Array(pql.UInt16(), size=4),
    "x_var": pql.Array(pql.UInt16(), size=2),
    "y": pql.UInt16(),
    "booleans": pql.Array(pql.Boolean(), size=3),
    "str_vals": pql.Array(pql.String(), size=3),
}
PL_SCHEMA = {
    "x": pl.Array(pl.UInt16, shape=4),
    "x_var": pl.Array(pl.UInt16, shape=2),
    "y": pl.UInt16,
    "booleans": pl.Array(pl.Boolean, shape=3),
    "str_vals": pl.Array(pl.String, shape=3),
}


def sample_df() -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    return nw.from_native(
        duckdb.from_arrow(
            pl.DataFrame(
                {
                    "x": [
                        [1, 2, 7, 3],
                        [3, 4, 5, 5],
                        [2, 5, 8, 1],
                        [1, 2, 3, 4],
                    ],
                    "x_var": [
                        [1, 2],
                        [1, 2],
                        [5, 2],
                        [1, 1],
                    ],
                    "y": [1, 2, 5, 0],
                    "booleans": [
                        [True, False, True],
                        [True, True, True],
                        [False, False, False],
                        [True, None, True],
                    ],
                    "str_vals": [
                        ["g", "b", "c"],
                        ["a", "b", "a"],
                        ["c", "c", "c"],
                        ["d", "e", "d"],
                    ],
                },
                schema=PL_SCHEMA,
            )
        )
    )


def assert_eq(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: nw.Expr | Iterable[nw.Expr]
) -> None:
    assert_frame_equal(
        pql.LazyFrame(sample_df().to_native()).cast(SCHEMA).select(pql_exprs).collect(),
        sample_df().lazy().select(polars_exprs).to_native().pl(),
        check_dtypes=False,
        check_row_order=False,
    )


def assert_eq_pl(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: pl.Expr | Iterable[pl.Expr]
) -> None:
    assert_frame_equal(
        pql.LazyFrame(sample_df().to_native()).cast(SCHEMA).select(pql_exprs).collect(),
        sample_df().to_native().pl(lazy=True).select(polars_exprs).collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def test_len() -> None:
    assert_eq(pql.col("x").arr.len(), nw.col("x").list.len())
    assert_eq_pl(pql.col("x_var").arr.len(), pl.col("x_var").arr.len())


def test_unique() -> None:
    assert_eq(pql.col("x").arr.unique(), nw.col("x").list.unique())


def test_n_unique() -> None:
    assert_eq_pl(pql.col("x").arr.n_unique(), pl.col("x").arr.n_unique())


def test_contains() -> None:
    assert_eq(pql.col("x").arr.contains(2), nw.col("x").list.contains(2))
    assert_eq_pl(
        (
            pql.col("x").arr.contains(pql.lit(None)).alias("x_nulls_neq"),
            pql.col("x").arr.contains("y").alias("x_nulls_neq_y"),
            pql.col("x").arr.contains(3).alias("x_y"),
        ),
        (
            pl.col("x")
            .arr.contains(pl.lit(None), nulls_equal=False)
            .alias("x_nulls_neq"),
            pl.col("x")
            .arr.contains(pl.col("y"), nulls_equal=False)
            .alias("x_nulls_neq_y"),
            pl.col("x").arr.contains(3).alias("x_y"),
        ),
    )


def test_count_matches() -> None:
    assert_eq_pl(pql.col("x").arr.count_matches(5), pl.col("x").arr.count_matches(5))


def test_drop_nulls() -> None:
    """Drop nulls don't exist for array in polars."""
    assert_eq_pl(
        pql.col("booleans").arr.drop_nulls(),
        pl.col("booleans").cast(pl.List(pl.Boolean)).list.drop_nulls(),
    )


def test_get() -> None:
    assert_eq(
        pql.col("x").arr.get(0),
        nw.col("x").list.get(0),
    )
    assert_eq_pl(
        pql.col("x").arr.get(-1),
        pl.col("x").arr.get(-1),
    )
    with pytest.raises(pl.exceptions.ComputeError, match="get index is out of bounds"):
        assert_eq_pl(
            pql.col("x_var").arr.get(10),
            pl.col("x_var").arr.get(10),
        )


def test_min() -> None:
    assert_eq(pql.col("x").arr.min(), nw.col("x").list.min())


def test_max() -> None:
    assert_eq(pql.col("x").arr.max(), nw.col("x").list.max())


def test_mean() -> None:
    assert_eq(pql.col("x").arr.mean(), nw.col("x").list.mean())


def test_median() -> None:
    assert_eq(pql.col("x").arr.median(), nw.col("x").list.median())


def test_sum() -> None:
    assert_eq(pql.col("x").arr.sum(), nw.col("x").list.sum())


def test_sort() -> None:
    assert_eq(
        (
            pql.col("x").arr.sort().alias("x_sorted"),
            pql.col("x")
            .arr.sort(descending=True, nulls_last=True)
            .alias("x_sorted_desc"),
        ),
        (
            nw.col("x").list.sort().alias("x_sorted"),
            nw.col("x")
            .list.sort(descending=True, nulls_last=True)
            .alias("x_sorted_desc"),
        ),
    )


def test_eval_num() -> None:
    assert_eq_pl(
        pql.col("x").arr.eval(pql.element().mul(2)),
        pl.col("x").arr.eval(pl.element().mul(2)),
    )


def test_eval_str() -> None:
    assert_eq_pl(
        pql.col("str_vals").arr.eval(pql.element().str.to_uppercase()),
        pl.col("str_vals").arr.eval(pl.element().str.to_uppercase()),
    )


def test_eval_bool() -> None:
    assert_eq_pl(
        pql.col("x").arr.eval(pql.element() > 3),
        pl.col("x").arr.eval(pl.element() > 3),
    )


def test_first() -> None:
    assert_eq_pl(pql.col("x").arr.first(), pl.col("x").arr.first())


def test_last() -> None:
    assert_eq_pl(pql.col("x").arr.last(), pl.col("x").arr.last())


def test_std() -> None:
    assert_eq_pl(pql.col("x").arr.std(), pl.col("x").arr.std())
    assert_eq_pl(pql.col("x").arr.std(ddof=0), pl.col("x").arr.std(ddof=0))


def test_var() -> None:
    assert_eq_pl(pql.col("x").arr.var(), pl.col("x").arr.var())
    assert_eq_pl(pql.col("x").arr.var(ddof=0), pl.col("x").arr.var(ddof=0))


def test_reverse() -> None:
    assert_eq_pl(pql.col("x").arr.reverse(), pl.col("x").arr.reverse())


def test_any() -> None:
    assert_eq_pl(pql.col("booleans").arr.any(), pl.col("booleans").arr.any())


def test_all() -> None:
    assert_eq_pl(pql.col("booleans").arr.all(), pl.col("booleans").arr.all())


def test_join() -> None:
    sep = pql.lit("-")
    assert_eq_pl(pql.col("str_vals").arr.join(sep), pl.col("str_vals").arr.join("-"))
    assert_eq_pl(
        pql.col("str_vals").arr.join(sep, ignore_nulls=False),
        pl.col("str_vals").arr.join("-", ignore_nulls=False),
    )


def test_filter() -> None:
    assert_eq_pl(
        pql.col("x").arr.filter(pql.element().gt(3)),
        pl.col("x").cast(pl.List(pl.UInt16)).list.filter(pl.element().gt(3)),
    )
