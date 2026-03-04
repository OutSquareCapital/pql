import narwhals as nw
import polars as pl
import pytest

import pql

from ._utils import assert_eq, assert_eq_pl


def test_all_fn() -> None:
    assert_eq(pql.all(), nw.all())


def test_len_fn() -> None:
    assert_eq(pql.len(), nw.len())


_MULTI_FNS = [
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "sum_horizontal",
    "min_horizontal",
    "max_horizontal",
    "mean_horizontal",
    "coalesce",
]


@pytest.mark.parametrize("fn", _MULTI_FNS)
def test_multi_col(fn: str) -> None:
    assert_eq(getattr(pql, fn)("x", "n"), getattr(nw, fn)("x", "n"))


def test_all_horizontal() -> None:
    assert_eq(
        pql.all_horizontal("a", "b"), nw.all_horizontal("a", "b", ignore_nulls=False)
    )


def test_any_horizontal() -> None:
    assert_eq(
        pql.any_horizontal("a", "b"), nw.any_horizontal("a", "b", ignore_nulls=False)
    )


def test_when_then_simple() -> None:
    pql_expr = (
        pql.when(pql.col("x").eq(5))
        .then(pql.lit("equal_to_5"))
        .otherwise(pql.lit("not_equal_to_5"))
    )
    pl_expr = (
        pl.when(pl.col("x").eq(5))
        .then(pl.lit("equal_to_5"))
        .otherwise(pl.lit("not_equal_to_5"))
    )
    assert_eq_pl(pql_expr, pl_expr)


def test_when_then_chained() -> None:
    pql_expr = (
        pql.when(pql.col("x") > 5)
        .then(pql.lit("high"))
        .when(pql.col("x") < 5)
        .then(pql.lit("low"))
        .when(pql.col("x") == 5)
        .then(pql.lit("equal"))
        .otherwise(pql.lit("mid"))
    )
    pl_expr = (
        pl.when(pl.col("x") > 5)
        .then(pl.lit("high"))
        .when(pl.col("x") < 5)
        .then(pl.lit("low"))
        .when(pl.col("x") == 5)
        .then(pl.lit("equal"))
        .otherwise(pl.lit("mid"))
    )
    assert_eq_pl(pql_expr, pl_expr)


def test_when_with_multiple_predicates() -> None:
    pql_expr = (
        pql.when(pql.col("a"), pql.col("b"))
        .then(pql.lit("both_true"))
        .otherwise(pql.lit("not_both_true"))
    )
    pl_expr = (
        pl.when(pl.col("a") & pl.col("b"))
        .then(pl.lit("both_true"))
        .otherwise(pl.lit("not_both_true"))
    )
    assert_eq_pl(pql_expr, pl_expr)


def test_when_without_otherwise() -> None:
    pql_expr = pql.when(pql.col("x") > 10).then(pql.lit("high"))
    pl_expr = pl.when(pl.col("x") > 10).then(pl.lit("high"))
    assert_eq_pl(pql_expr, pl_expr)


def test_when_nested_conditions() -> None:
    pql_expr = (
        pql.when(pql.col("x") > 15)
        .then(
            pql.when(pql.col("age") > 30)
            .then(pql.lit("x_high_age_high"))
            .otherwise(pql.lit("x_high_age_low"))
        )
        .otherwise(pql.lit("x_low"))
    )
    pl_expr = (
        pl.when(pl.col("x") > 15)
        .then(
            pl.when(pl.col("age") > 30)
            .then(pl.lit("x_high_age_high"))
            .otherwise(pl.lit("x_high_age_low"))
        )
        .otherwise(pl.lit("x_low"))
    )
    assert_eq_pl(pql_expr, pl_expr)
