import narwhals as nw
import polars as pl

import pql

from ._utils import assert_eq, assert_eq_pl


def test_all_fn() -> None:
    assert_eq(pql.all(), nw.all())


def test_sum_horizontal() -> None:
    assert_eq(pql.sum_horizontal("x", "n"), nw.sum_horizontal("x", "n"))


def test_sum_horizontal_iterable() -> None:
    assert_eq(pql.sum_horizontal(("x", "n")), nw.sum_horizontal(("x", "n")))


def test_min_horizontal() -> None:
    assert_eq(pql.min_horizontal("x", "n"), nw.min_horizontal("x", "n"))


def test_max_horizontal() -> None:
    assert_eq(pql.max_horizontal("x", "n"), nw.max_horizontal("x", "n"))


def test_mean_horizontal() -> None:
    assert_eq(pql.mean_horizontal("x", "n"), nw.mean_horizontal("x", "n"))


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
