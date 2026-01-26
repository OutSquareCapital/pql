"""Tests for expression operations to achieve 100% coverage - comparing with Polars."""

from __future__ import annotations

from datetime import UTC, date, datetime
from functools import partial

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pql

assert_eq = partial(assert_frame_equal, check_dtypes=False, check_row_order=False)


assert_eq = partial(assert_frame_equal, check_dtypes=False, check_row_order=False)


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "sex": ["F", "M", "M", "M", "F"],
            "age": [25, 30, 35, 28, 22],
            "salary": [50000.0, 60000.0, 75000.0, 55000.0, 45000.0],
            "department": [
                "Engineering",
                "Sales",
                "Engineering",
                "Sales",
                "Engineering",
            ],
            "is_active": [True, True, False, True, True],
        }
    )


@pytest.fixture
def sample_df_with_nulls() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10.0, None, 30.0, None, 50.0],
            "category": ["A", "B", None, "A", "B"],
        }
    )


@pytest.fixture
def sample_df_strings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "text": [
                "Hello World",
                "hello",
                "HELLO",
                "  trimmed  ",
                "prefix_test_suffix",
            ],
            "pattern": ["World", "lo", "LL", "trimmed", "test"],
        }
    )


@pytest.fixture
def sample_df_dates() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC),
                datetime(2024, 6, 20, 14, 0, 0, tzinfo=UTC),
                datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC),
            ],
            "dt": [
                date(2024, 1, 15),
                date(2024, 6, 20),
                date(2024, 12, 31),
            ],
        }
    )


@pytest.fixture
def orders_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "customer_id": [1, 1, 2, 2, 3, 3, 3, 1],
            "amount": [100.0, 150.0, 200.0, 50.0, 300.0, 100.0, 250.0, 75.0],
            "status": [
                "completed",
                "completed",
                "pending",
                "completed",
                "completed",
                "cancelled",
                "completed",
                "pending",
            ],
        }
    )


@pytest.fixture
def left_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "left_value": ["a", "b", "c", "d"],
        }
    )


@pytest.fixture
def right_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [2, 3, 4, 5],
            "right_value": ["x", "y", "z", "w"],
        }
    )


def test_select_single_column(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).select("name").collect(),
        sample_df.lazy().select("name").collect(),
    )


def test_select_multiple_columns(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).select("name", "age", "salary").collect(),
        sample_df.lazy().select("name", "age", "salary").collect(),
    )


def test_select_with_expression(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .select(pql.col("name"), pql.col("salary").mul(1.1).alias("salary_increase"))
        .collect(),
        sample_df.lazy()
        .select(
            pl.col("name"),
            pl.col("salary").mul(1.1).alias("salary_increase"),
        )
        .collect(),
    )


def test_filter_gt(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).filter(pql.col("age").gt(25)).collect(),
        sample_df.lazy().filter(pl.col("age").gt(25)).collect(),
    )


def test_filter_eq_string(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .filter(pql.col("department").eq("Engineering"))
        .collect(),
        sample_df.lazy().filter(pl.col("department").eq("Engineering")).collect(),
    )


def test_filter_combined_and(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .filter(pql.col("age").gt(25).and_(pql.col("salary").ge(55000)))
        .collect(),
        sample_df.lazy()
        .filter(pl.col("age").gt(25).and_(pl.col("salary").ge(55000)))
        .collect(),
    )


def test_filter_combined_or(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .filter((pql.col("age").lt(25)).or_(pql.col("salary").gt(70000)))
        .collect(),
        sample_df.lazy()
        .filter((pl.col("age").lt(25)).or_(pl.col("salary").gt(70000)))
        .collect(),
    )


def test_filter_not(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).filter(pql.col("is_active").not_()).collect(),
        sample_df.lazy().filter(pl.col("is_active").not_()).collect(),
    )


def test_filter_is_in(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .filter(pql.col("name").is_in(["Alice", "Bob", "Charlie"]))
        .collect(),
        sample_df.lazy()
        .filter(pl.col("name").is_in(["Alice", "Bob", "Charlie"]))
        .collect(),
    )


def test_filter_is_null(sample_df_with_nulls: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_with_nulls)
        .filter(pql.col("value").is_null())
        .collect(),
        sample_df_with_nulls.lazy().filter(pl.col("value").is_null()).collect(),
    )


def test_filter_is_not_null(sample_df_with_nulls: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_with_nulls)
        .filter(pql.col("value").is_not_null())
        .collect(),
        sample_df_with_nulls.lazy().filter(pl.col("value").is_not_null()).collect(),
    )


def test_fill_null(sample_df_with_nulls: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_with_nulls)
        .select(
            pql.col("id"),
            pql.col("value").fill_null(0.0).alias("value"),
        )
        .collect(),
        sample_df_with_nulls.lazy()
        .select(
            pl.col("id"),
            pl.col("value").fill_null(0.0),
        )
        .collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df_with_nulls)
        .select(pql.col("id"), pql.col("value"))
        .fill_null(0)
        .collect(),
        (
            sample_df_with_nulls.lazy()
            .select(pl.col("id"), pl.col("value"))
            .fill_null(0)
            .collect()
        ),
    )

    assert_eq(
        pql.LazyFrame(sample_df_with_nulls).fill_null(strategy="forward").collect(),
        sample_df_with_nulls.lazy().fill_null(strategy="forward").collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df_with_nulls).fill_null(strategy="backward").collect(),
        sample_df_with_nulls.lazy().fill_null(strategy="backward").collect(),
    )
    with pytest.raises(
        ValueError, match=r"Either `value` or `strategy` must be provided\."
    ):
        pql.LazyFrame(sample_df_with_nulls).fill_null().collect()


def test_sort(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).sort("age").collect(),
        sample_df.lazy().sort("age").collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df).sort("salary", descending=True).collect(),
        sample_df.lazy().sort("salary", descending=True).collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df)
        .sort("department", "age", descending=[False, True])
        .collect(),
        sample_df.lazy().sort("department", "age", descending=[False, True]).collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df)
        .sort("department", "age", descending=True, nulls_last=True)
        .collect(),
        sample_df.lazy()
        .sort("department", "age", descending=True, nulls_last=True)
        .collect(),
    )


def test_sort_with_nulls_last() -> None:
    df = pl.DataFrame({"a": [1, None, 2, None, 3], "b": [5, 4, 3, 2, 1]})
    result = pql.LazyFrame(df).sort("a", nulls_last=True).collect()
    expected = df.lazy().sort("a", nulls_last=True).collect()
    assert_eq(result, expected)


def test_sort_with_nulls_last_multiple() -> None:
    df = pl.DataFrame({"a": [1, None, 2], "b": [None, 4, 5]})
    result = pql.LazyFrame(df).sort("a", "b", nulls_last=[True, False]).collect()
    expected = df.lazy().sort("a", "b", nulls_last=[True, False]).collect()
    assert_eq(result, expected)


def test_sort_by_descending() -> None:
    df = pl.DataFrame({"name": ["Charlie", "Alice", "Bob"], "age": [35, 25, 30]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("name").sort_by("age", descending=True))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("name").sort_by("age", descending=True))
        .collect(),
    )


def test_limit(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).sort("id").limit(3).collect(),
        sample_df.lazy().sort("id").limit(3).collect(),
    )


def test_head(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).sort("id").head(2).collect(),
        sample_df.lazy().sort("id").head(2).collect(),
    )


def test_add(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .select(
            pql.col("age").add(10).alias("age_plus_10"),
            (pql.col("age") + 10).alias("age_plus_10_bis"),
        )
        .collect(),
        sample_df.lazy()
        .select(
            pl.col("age").add(10).alias("age_plus_10"),
            (pl.col("age") + 10).alias("age_plus_10_bis"),
        )
        .collect(),
    )


def test_multiply(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .select(
            pql.col("id"),
            pql.col("salary").mul(2).alias("double_salary"),
        )
        .collect(),
        sample_df.lazy()
        .select(
            pl.col("id"),
            pl.col("salary").mul(2).alias("double_salary"),
        )
        .collect(),
    )


def test_divide(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .select(
            pql.col("id"),
            pql.col("salary").truediv(1000).alias("salary_k"),
        )
        .collect(),
        sample_df.lazy()
        .select(
            pl.col("id"),
            pl.col("salary").truediv(1000).alias("salary_k"),
        )
        .collect(),
    )


def test_mod(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .select(
            (pql.col("age") % 10).alias("age_mod_10_bis"),
            pql.col("age").mod(10).alias("age_mod_10"),
        )
        .collect(),
        sample_df.lazy()
        .select(
            (pl.col("age") % 10).alias("age_mod_10_bis"),
            pl.col("age").mod(10).alias("age_mod_10"),
        )
        .collect(),
    )


def test_abs() -> None:
    abs_df = pl.DataFrame({"x": [-5, -3, 0, 3, 5]})
    assert_eq(
        pql.LazyFrame(abs_df)
        .select(
            pql.col("x"),
            pql.col("x").abs().alias("abs_x"),
        )
        .collect(),
        abs_df.lazy()
        .select(
            pl.col("x"),
            pl.col("x").abs().alias("abs_x"),
        )
        .collect(),
    )


def test_with_columns_filter(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .select(pql.col("salary").mul(12).alias("annual_salary"))
        .filter(pql.col("annual_salary").gt(600000))
        .collect(),
        sample_df.lazy()
        .select(pl.col("salary").mul(12).alias("annual_salary"))
        .filter(pl.col("annual_salary").gt(600000))
        .collect(),
    )


def test_reverse(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).reverse().collect(),
        sample_df.lazy().reverse().collect(),
    )


def test_first(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).first().collect(), sample_df.lazy().first().collect()
    )


def test_count(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("id")).count().collect()
    expected = sample_df.lazy().select(pl.col("id")).count().collect()
    assert_eq(result, expected, check_dtypes=False)


def test_sum(sample_df: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df)
        .select(pql.col("age"), pql.col("salary"))
        .sum()
        .collect()
    )
    expected = sample_df.lazy().select(pl.col("age"), pl.col("salary")).sum().collect()
    assert_eq(result, expected, check_dtypes=False)


def test_mean(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age")).mean().collect()
    expected = sample_df.lazy().select(pl.col("age")).mean().collect()
    assert_eq(result, expected)


def test_median(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("salary")).median().collect()
    expected = sample_df.lazy().select(pl.col("salary")).median().collect()
    assert_eq(result, expected)


def test_min(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age")).min().collect()
    expected = sample_df.lazy().select(pl.col("age")).min().collect()
    assert_eq(result, expected)


def test_max(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age")).max().collect()
    expected = sample_df.lazy().select(pl.col("age")).max().collect()
    assert_eq(result, expected)


def test_null_count(sample_df_with_nulls: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df_with_nulls)
        .select(pql.col("value"))
        .null_count()
        .collect()
    )
    expected = (
        sample_df_with_nulls.lazy().select(pl.col("value")).null_count().collect()
    )
    assert_eq(result, expected, check_dtypes=False)


def test_drop_nulls(sample_df_with_nulls: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df_with_nulls).drop_nulls().collect()
    expected = sample_df_with_nulls.lazy().drop_nulls().collect()
    assert_eq(result, expected)


def test_drop_nulls_subset(sample_df_with_nulls: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df_with_nulls).drop_nulls(subset="value").collect()
    expected = sample_df_with_nulls.lazy().drop_nulls(subset="value").collect()
    assert_eq(result, expected)


def test_with_row_index(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).with_row_index().collect()
    expected = sample_df.lazy().with_row_index().collect()
    assert_eq(result, expected, check_dtypes=False)


def test_with_row_index_custom_name(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).with_row_index(name="row_num").collect()
    expected = sample_df.lazy().with_row_index(name="row_num").collect()
    assert_eq(result, expected, check_dtypes=False)


def test_with_row_index_offset(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).with_row_index(offset=10).collect()
    expected = sample_df.lazy().with_row_index(offset=10).collect()
    assert_eq(result, expected, check_dtypes=False)


def test_quantile(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age")).quantile(0.5).collect()
    expected = sample_df.lazy().select(pl.col("age")).quantile(0.5).collect()
    assert_eq(result, expected)


def test_gather_every(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).gather_every(2).collect()
    expected = sample_df.lazy().gather_every(2).collect()
    assert_eq(result, expected)


def test_gather_every_with_offset(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).gather_every(2, offset=1).collect()
    expected = sample_df.lazy().gather_every(2, offset=1).collect()
    assert_eq(result, expected)


def test_clear(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).clear().collect()
    expected = sample_df.lazy().clear().collect()
    assert_eq(result, expected)


def test_top_k(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).top_k(3, by="age").collect()
    expected = sample_df.lazy().top_k(3, by="age").collect()
    assert_eq(result, expected)


def test_bottom_k(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).bottom_k(3, by="age").collect()
    expected = sample_df.lazy().bottom_k(3, by="age").collect()
    assert_eq(result, expected)


def test_cast_single_dtype(sample_df: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df)
        .select(pql.col("age"))
        .cast(pql.datatypes.Float64)
        .collect()
    )
    expected = sample_df.lazy().select(pl.col("age")).cast(pl.Float64).collect()
    assert_frame_equal(result, expected)


def test_cast_mapping(sample_df: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df)
        .select(pql.col("age"), pql.col("id"))
        .cast({"age": pql.datatypes.Float64})
        .collect()
    )
    expected = (
        sample_df.lazy()
        .select(pl.col("age"), pl.col("id"))
        .cast({"age": pl.Float64})
        .collect()
    )
    assert_frame_equal(result, expected)


def test_pipe(sample_df: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df)
        .pipe(lambda lf: lf.with_columns(pql.lit(1).alias("new_col")))
        .collect()
    )
    expected = (
        sample_df.lazy()
        .pipe(lambda df: df.with_columns(pl.lit(1).alias("new_col")))
        .collect()
    )
    assert_eq(result, expected)


def test_pow() -> None:
    math_df = pl.DataFrame({"x": [2, 3, 4]})
    result = (
        pql.LazyFrame(math_df)
        .select(
            pql.col("x").pow(2).alias("x_squared"),
            (pql.col("x") ** 2).alias("x_squared_bis"),
        )
        .collect()
    )
    expected = math_df.lazy().select(
        pl.col("x").pow(2).alias("x_squared"), (pl.col("x") ** 2).alias("x_squared_bis")
    )
    assert_eq(result, expected)


def test_floor() -> None:
    math_df = pl.DataFrame({"x": [2.0, 3.3, 4.9]})
    result = (
        pql.LazyFrame(math_df).select(pql.col("x").floor().alias("x_floor")).collect()
    )
    expected = math_df.lazy().select(pl.col("x").floor().alias("x_floor")).collect()
    assert_eq(result, expected)


def test_ceil() -> None:
    math_df = pl.DataFrame({"x": [2.1, 3.3, 4.9]})
    result = (
        pql.LazyFrame(math_df).select(pql.col("x").ceil().alias("x_ceil")).collect()
    )
    expected = math_df.lazy().select(pl.col("x").ceil().alias("x_ceil")).collect()
    assert_eq(result, expected)


def test_round() -> None:
    math_df = pl.DataFrame({"x": [2.156, 3.345, 4.891]})
    result = (
        pql.LazyFrame(math_df).select(pql.col("x").round(2).alias("x_round")).collect()
    )
    expected = math_df.lazy().select(pl.col("x").round(2).alias("x_round")).collect()
    assert_eq(result, expected)


def test_sqrt() -> None:
    math_df = pl.DataFrame({"x": [4.0, 9.0, 16.0]})
    result = (
        pql.LazyFrame(math_df).select(pql.col("x").sqrt().alias("x_sqrt")).collect()
    )
    expected = math_df.lazy().select(pl.col("x").sqrt().alias("x_sqrt")).collect()
    assert_eq(result, expected)


def test_cbrt() -> None:
    math_df = pl.DataFrame({"x": [8.0, 27.0, 64.0]})
    result = (
        pql.LazyFrame(math_df).select(pql.col("x").cbrt().alias("x_cbrt")).collect()
    )
    expected = math_df.lazy().select(pl.col("x").cbrt().alias("x_cbrt")).collect()
    assert_eq(result, expected)


def test_log() -> None:
    math_df = pl.DataFrame({"x": [1.0, 10.0, 100.0]})
    result = (
        pql.LazyFrame(math_df).select(pql.col("x").log(10).alias("x_log10")).collect()
    )
    expected = math_df.lazy().select(pl.col("x").log(10).alias("x_log10")).collect()
    assert_eq(result, expected)


def test_log10() -> None:
    math_df = pl.DataFrame({"x": [1.0, 10.0, 100.0]})
    result = (
        pql.LazyFrame(math_df).select(pql.col("x").log10().alias("x_log10")).collect()
    )
    expected = math_df.lazy().select(pl.col("x").log10().alias("x_log10")).collect()
    assert_eq(result, expected)


def test_log1p() -> None:
    math_df = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
    result = (
        pql.LazyFrame(math_df).select(pql.col("x").log1p().alias("x_log1p")).collect()
    )
    expected = math_df.lazy().select(pl.col("x").log1p().alias("x_log1p")).collect()
    assert_eq(result, expected)


def test_exp() -> None:
    math_df = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
    result = pql.LazyFrame(math_df).select(pql.col("x").exp().alias("x_exp")).collect()
    expected = math_df.lazy().select(pl.col("x").exp().alias("x_exp")).collect()
    assert_eq(result, expected)


def test_sin() -> None:
    trig_df = pl.DataFrame({"x": [0.0, 1.5708, 3.14159]})
    result = pql.LazyFrame(trig_df).select(pql.col("x").sin().alias("sin_x")).collect()
    expected = trig_df.lazy().select(pl.col("x").sin().alias("sin_x")).collect()
    assert_eq(result, expected)


def test_cos() -> None:
    trig_df = pl.DataFrame({"x": [0.0, 1.5708, 3.14159]})
    result = pql.LazyFrame(trig_df).select(pql.col("x").cos().alias("cos_x")).collect()
    expected = trig_df.lazy().select(pl.col("x").cos().alias("cos_x")).collect()
    assert_eq(result, expected)


def test_tan() -> None:
    trig_df = pl.DataFrame({"x": [0.0, 0.785, 1.0]})
    result = pql.LazyFrame(trig_df).select(pql.col("x").tan().alias("tan_x")).collect()
    expected = trig_df.lazy().select(pl.col("x").tan().alias("tan_x")).collect()
    assert_eq(result, expected)


def test_arcsin() -> None:
    trig_df = pl.DataFrame({"x": [0.0, 0.5, 1.0]})
    result = (
        pql.LazyFrame(trig_df).select(pql.col("x").arcsin().alias("arcsin_x")).collect()
    )
    expected = trig_df.lazy().select(pl.col("x").arcsin().alias("arcsin_x")).collect()
    assert_eq(result, expected)


def test_arccos() -> None:
    trig_df = pl.DataFrame({"x": [0.0, 0.5, 1.0]})
    result = (
        pql.LazyFrame(trig_df).select(pql.col("x").arccos().alias("arccos_x")).collect()
    )
    expected = trig_df.lazy().select(pl.col("x").arccos().alias("arccos_x")).collect()
    assert_eq(result, expected)


def test_arctan() -> None:
    trig_df = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
    result = (
        pql.LazyFrame(trig_df).select(pql.col("x").arctan().alias("arctan_x")).collect()
    )
    expected = trig_df.lazy().select(pl.col("x").arctan().alias("arctan_x")).collect()
    assert_eq(result, expected)


def test_degrees() -> None:
    trig_df = pl.DataFrame({"x": [0.0, 3.14159, 6.28318]})
    result = (
        pql.LazyFrame(trig_df)
        .select(pql.col("x").degrees().alias("x_degrees"))
        .collect()
    )
    expected = trig_df.lazy().select(pl.col("x").degrees().alias("x_degrees")).collect()
    assert_eq(result, expected)


def test_radians() -> None:
    trig_df = pl.DataFrame({"x": [0.0, 90.0, 180.0]})
    result = (
        pql.LazyFrame(trig_df)
        .select(pql.col("x").radians().alias("x_radians"))
        .collect()
    )
    expected = trig_df.lazy().select(pl.col("x").radians().alias("x_radians")).collect()
    assert_eq(result, expected)


def test_sign() -> None:
    sign_df = pl.DataFrame({"x": [-5.0, 0.0, 5.0]})
    result = (
        pql.LazyFrame(sign_df).select(pql.col("x").sign().alias("sign_x")).collect()
    )
    expected = sign_df.lazy().select(pl.col("x").sign().alias("sign_x")).collect()
    assert_eq(result, expected)


def test_clip() -> None:
    clip_df = pl.DataFrame({"x": [1.0, 5.0, 10.0, 15.0, 20.0]})
    result = (
        pql.LazyFrame(clip_df)
        .select(pql.col("x").clip(5.0, 15.0).alias("x_clipped"))
        .collect()
    )
    expected = (
        clip_df.lazy().select(pl.col("x").clip(5.0, 15.0).alias("x_clipped")).collect()
    )
    assert_eq(result, expected)


def test_drop_single_column(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).drop("age").collect()
    expected = sample_df.lazy().drop("age").collect()
    assert_eq(result, expected)


def test_drop_multiple_columns(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).drop("age", "salary").collect()
    expected = sample_df.lazy().drop("age", "salary").collect()
    assert_eq(result, expected)


def test_rename_single_column(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).rename({"age": "years"}).collect()
    expected = sample_df.lazy().rename({"age": "years"}).collect()
    assert_eq(result, expected)


def test_rename_multiple_columns(sample_df: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df).rename({"age": "years", "name": "full_name"}).collect()
    )
    expected = sample_df.lazy().rename({"age": "years", "name": "full_name"}).collect()
    assert_eq(result, expected)


def test_with_columns_single_expr(sample_df: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df)
        .with_columns(pql.col("age").mul(2).alias("age_doubled"))
        .collect()
    )
    expected = (
        sample_df.lazy()
        .with_columns(pl.col("age").mul(2).alias("age_doubled"))
        .collect()
    )
    assert_eq(result, expected)


def test_with_columns_multiple_exprs(sample_df: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df)
        .with_columns(
            pql.col("age").mul(2).alias("age_doubled"),
            pql.col("salary").truediv(12).alias("monthly_salary"),
        )
        .collect()
    )
    expected = (
        sample_df.lazy()
        .with_columns(
            (pl.col("age") * 2).alias("age_doubled"),
            (pl.col("salary") / 12).alias("monthly_salary"),
        )
        .collect()
    )
    assert_eq(result, expected)


def test_std_default(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select("age").std().collect()
    expected = sample_df.lazy().select("age").std().collect()
    assert_eq(result, expected)


def test_var_default(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select("age").var().collect()
    expected = sample_df.lazy().select("age").var().collect()
    assert_eq(result, expected)


def test_fill_nan_with_value() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan"), 3.0, float("nan"), 5.0]})
    result = pql.LazyFrame(df).fill_nan(0.0).collect()
    expected = df.lazy().fill_nan(0.0).collect()
    assert_eq(result, expected)


def test_drop_nans() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan"), 3.0], "b": [4.0, 5.0, float("nan")]})
    result = pql.LazyFrame(df).drop_nans().collect()
    expected = df.lazy().drop_nans().collect()
    assert_eq(result, expected)


def test_drop_nans_subset() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan"), 3.0], "b": [4.0, 5.0, float("nan")]})
    result = pql.LazyFrame(df).drop_nans(subset="a").collect()
    expected = df.lazy().drop_nans(subset="a").collect()
    assert_eq(result, expected)


def test_shift_positive() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = pql.LazyFrame(df).shift(2).collect()
    expected = df.lazy().shift(2).collect()
    assert_eq(result, expected)


def test_shift_negative() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = pql.LazyFrame(df).shift(-2).collect()
    expected = df.lazy().shift(-2).collect()
    assert_eq(result, expected)


def test_shift_with_fill_value() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = pql.LazyFrame(df).shift(2, fill_value=0).collect()
    expected = df.lazy().shift(2, fill_value=0).collect()
    assert_eq(result, expected)


def test_unique_with_subset() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    result = pql.LazyFrame(df).unique(subset="a").collect()
    expected = df.lazy().unique(subset="a").collect()
    assert_eq(result, expected)


def test_cum_sum() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").cum_sum().alias("cum_sum"))
        .collect()
    )
    expected = df.lazy().with_columns(pl.col("a").cum_sum().alias("cum_sum")).collect()
    assert_eq(result, expected)


def test_cum_max() -> None:
    df = pl.DataFrame({"a": [1, 5, 3, 4, 2]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").cum_max().alias("cum_max"))
        .collect()
    )
    expected = df.lazy().with_columns(pl.col("a").cum_max().alias("cum_max")).collect()
    assert_eq(result, expected)


def test_cum_min() -> None:
    df = pl.DataFrame({"a": [5, 2, 3, 1, 4]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").cum_min().alias("cum_min"))
        .collect()
    )
    expected = df.lazy().with_columns(pl.col("a").cum_min().alias("cum_min")).collect()
    assert_eq(result, expected)


def test_cum_count() -> None:
    df = pl.DataFrame({"a": [1, 2, None, 4, 5]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").cum_count().alias("cum_count"))
        .collect()
    )
    expected = (
        df.lazy().with_columns(pl.col("a").cum_count().alias("cum_count")).collect()
    )
    assert_eq(result, expected)


def test_cum_prod() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").cum_prod().alias("cum_prod"))
        .collect()
    )
    expected = (
        df.lazy().with_columns(pl.col("a").cum_prod().alias("cum_prod")).collect()
    )
    assert_eq(result, expected)


def test_forward_fill() -> None:
    df = pl.DataFrame({"a": [1, None, 3, None, 5]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").forward_fill().alias("forward_filled"))
        .collect()
    )
    expected = (
        df.lazy()
        .with_columns(pl.col("a").forward_fill().alias("forward_filled"))
        .collect()
    )
    assert_eq(result, expected)


def test_backward_fill() -> None:
    df = pl.DataFrame({"a": [1, None, 3, None, 5]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").backward_fill().alias("backward_filled"))
        .collect()
    )
    expected = (
        df.lazy()
        .with_columns(pl.col("a").backward_fill().alias("backward_filled"))
        .collect()
    )
    assert_eq(result, expected)


def test_explode() -> None:
    df = pl.DataFrame({"a": [[1, 2], [3, 4, 5]]})
    result = (
        pql.LazyFrame(df).select(pql.col("a").explode().alias("exploded")).collect()
    )
    expected = df.lazy().select(pl.col("a").explode().alias("exploded")).collect()
    assert_eq(result, expected)


def test_flatten() -> None:
    df = pl.DataFrame({"a": [[[1, 2]], [[3, 4, 5]]]})
    result = (
        pql.LazyFrame(df).select(pql.col("a").flatten().alias("flattened")).collect()
    )
    expected = df.lazy().select(pl.col("a").flatten().alias("flattened")).collect()
    assert_eq(result, expected)


def test_col_interpolate() -> None:
    df = pl.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
    result = (
        pql.LazyFrame(df)
        .select(pql.col("x").interpolate().alias("interpolated"))
        .collect()
    )
    expected = (
        df.lazy().select(pl.col("x").interpolate().alias("interpolated")).collect()
    )
    assert_eq(result, expected)


def test_col_sample() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    result = pql.LazyFrame(df).select(pql.col("x").sample(n=2)).collect()
    assert len(result) > 0


def test_col_shuffle() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    result = pql.LazyFrame(df).select(pql.col("x").shuffle()).collect()
    assert len(result) == 5


def test_col_explode() -> None:
    df = pl.DataFrame({"x": [[1, 2], [3, 4, 5]]})
    result = (
        pql.LazyFrame(df).select(pql.col("x").explode().alias("exploded")).collect()
    )
    expected = df.lazy().select(pl.col("x").explode().alias("exploded")).collect()
    assert_eq(result, expected)


def test_col_reverse() -> None:
    df = pl.DataFrame({"x": [[1, 2, 3], [4, 5, 6]]})
    result = (
        pql.LazyFrame(df).select(pql.col("x").reverse().alias("reversed")).collect()
    )
    expected = df.lazy().select(pl.col("x").reverse().alias("reversed")).collect()
    assert_eq(result, expected)


def test_is_nan() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan"), 3.0]})
    result = (
        pql.LazyFrame(df).with_columns(pql.col("a").is_nan().alias("is_nan")).collect()
    )
    expected = df.lazy().with_columns(pl.col("a").is_nan().alias("is_nan")).collect()
    assert_eq(result, expected)


def test_is_not_nan() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan"), 3.0]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").is_not_nan().alias("is_not_nan"))
        .collect()
    )
    expected = (
        df.lazy().with_columns(pl.col("a").is_not_nan().alias("is_not_nan")).collect()
    )
    assert_eq(result, expected)


def test_is_finite() -> None:
    df = pl.DataFrame({"a": [1.0, float("inf"), float("nan")]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").is_finite().alias("is_finite"))
        .collect()
    )
    expected = (
        df.lazy().with_columns(pl.col("a").is_finite().alias("is_finite")).collect()
    )
    assert_eq(result, expected)


def test_is_infinite() -> None:
    df = pl.DataFrame({"a": [1.0, float("inf"), float("-inf")]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").is_infinite().alias("is_infinite"))
        .collect()
    )
    expected = (
        df.lazy().with_columns(pl.col("a").is_infinite().alias("is_infinite")).collect()
    )
    assert_eq(result, expected)


def test_fill_nan_expr() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan"), 3.0]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").fill_nan(0.0).alias("filled"))
        .collect()
    )
    expected = (
        df.lazy().with_columns(pl.col("a").fill_nan(0.0).alias("filled")).collect()
    )
    assert_eq(result, expected)


def test_is_duplicated() -> None:
    df = pl.DataFrame({"a": [1, 2, 2, 3]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").is_duplicated().alias("is_dup"))
        .collect()
    )
    expected = (
        df.lazy().with_columns(pl.col("a").is_duplicated().alias("is_dup")).collect()
    )
    assert_eq(result, expected)


def test_is_unique() -> None:
    df = pl.DataFrame({"a": [1, 2, 2, 3]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").is_unique().alias("is_unique"))
        .collect()
    )
    expected = (
        df.lazy().with_columns(pl.col("a").is_unique().alias("is_unique")).collect()
    )
    assert_eq(result, expected)


def test_is_first_distinct() -> None:
    df = pl.DataFrame({"a": [1, 2, 2, 3, 3, 3]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").is_first_distinct().alias("is_first"))
        .collect()
    )
    expected = (
        df.lazy()
        .with_columns(pl.col("a").is_first_distinct().alias("is_first"))
        .collect()
    )
    assert_eq(result, expected)


def test_is_last_distinct() -> None:
    df = pl.DataFrame({"a": [1, 2, 2, 3, 3, 3]})
    result = (
        pql.LazyFrame(df)
        .with_columns(pql.col("a").is_last_distinct().alias("is_last"))
        .collect()
    )
    expected = (
        df.lazy()
        .with_columns(pl.col("a").is_last_distinct().alias("is_last"))
        .collect()
    )
    assert_eq(result, expected)


def test_sinh() -> None:
    df = pl.DataFrame({"a": [0.0, 1.0, 2.0]})
    result = pql.LazyFrame(df).select(pql.col("a").sinh().alias("sinh")).collect()
    expected = df.lazy().select(pl.col("a").sinh().alias("sinh")).collect()
    assert_eq(result, expected)


def test_cosh() -> None:
    df = pl.DataFrame({"a": [0.0, 1.0, 2.0]})
    result = pql.LazyFrame(df).select(pql.col("a").cosh().alias("cosh")).collect()
    expected = df.lazy().select(pl.col("a").cosh().alias("cosh")).collect()
    assert_eq(result, expected)


def test_tanh() -> None:
    df = pl.DataFrame({"a": [0.0, 1.0, 2.0]})
    result = pql.LazyFrame(df).select(pql.col("a").tanh().alias("tanh")).collect()
    expected = df.lazy().select(pl.col("a").tanh().alias("tanh")).collect()
    assert_eq(result, expected)


def test_arithmetic_operators() -> None:
    df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").neg().alias("neg")).collect(),
        df.lazy().select((-pl.col("x")).alias("neg")).collect(),
    )


def test_col_getattr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col.a).collect(),
        df.lazy().select(pl.col.a).collect(),
    )


def test_all_expr() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert_eq(
        pql.LazyFrame(df).select(pql.all()).collect(),
        df.lazy().select(pl.all()).collect(),
    )


def test_round_modes() -> None:
    df = pl.DataFrame({"x": [1.5, 2.5, 3.5]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").round(0, mode="half_to_even").alias("rounded"))
        .collect(),
        df.lazy().select(pl.col("x").round(0).alias("rounded")).collect(),
    )
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").round(0, mode="round").alias("rounded"))
        .collect(),
        df.lazy().select(pl.col("x").round_sig_figs(1).alias("rounded")).collect(),
    )


def test_clip_bounds() -> None:
    df = pl.DataFrame({"x": [1, 5, 10, 15, 20]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").clip(lower_bound=5).alias("clipped"))
        .collect(),
        df.lazy().select(pl.col("x").clip(5).alias("clipped")).collect(),
    )
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").clip(upper_bound=15).alias("clipped"))
        .collect(),
        df.lazy().select(pl.col("x").clip(upper_bound=15).alias("clipped")).collect(),
    )


def test_cum_count_reverse() -> None:
    df = pl.DataFrame({"a": [1, 2, None, 4]})
    assert_eq(
        pql.LazyFrame(df)
        .with_columns(pql.col("a").cum_count(reverse=True).alias("cum_count"))
        .collect(),
        df.lazy()
        .with_columns(pl.col("a").cum_count(reverse=True).alias("cum_count"))
        .collect(),
    )


def test_pipe_expr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("a").pipe(lambda x: x * 2).alias("doubled"))
        .collect(),
        df.lazy()
        .select((pl.col("a").pipe(lambda x: x * 2)).alias("doubled"))
        .collect(),
    )


def test_unique_no_subset() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    assert_eq(pql.LazyFrame(df).unique().collect(), df.lazy().unique().collect())


def test_std_var_ddof() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert_eq(
        pql.LazyFrame(df).select("a").std(ddof=0).collect(),
        df.lazy().select("a").std(ddof=0).collect(),
    )
    assert_eq(
        pql.LazyFrame(df).select("a").var(ddof=0).collect(),
        df.lazy().select("a").var(ddof=0).collect(),
    )


def test_interpolate(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).interpolate().collect()
    expected = sample_df.lazy().interpolate().collect()
    assert_eq(result, expected)


def test_clear_with_n(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).clear(n=3).collect()
    expected = sample_df.lazy().clear(n=3).collect()
    assert_eq(result, expected)


def test_shift_with_expr_fill() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = pql.LazyFrame(df).shift(2, fill_value=999).collect()
    expected = df.lazy().shift(2, fill_value=999).collect()
    assert_eq(result, expected)


def test_top_k_with_multiple_cols(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).top_k(3, by=["department", "age"]).collect()
    expected = sample_df.lazy().top_k(3, by=["department", "age"]).collect()
    assert_eq(result, expected)


def test_top_k_with_reverse(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).top_k(2, by="age", reverse=True).collect()
    expected = sample_df.lazy().top_k(2, by="age", reverse=True).collect()
    assert_eq(result, expected)


def test_bottom_k_with_multiple_cols(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).bottom_k(3, by=["department", "age"]).collect()
    expected = sample_df.lazy().bottom_k(3, by=["department", "age"]).collect()
    assert_eq(result, expected)


def test_bottom_k_with_reverse(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).bottom_k(2, by="age", reverse=True).collect()
    expected = sample_df.lazy().bottom_k(2, by="age", reverse=True).collect()
    assert_eq(result, expected)


def test_rsub() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    assert_eq(
        pql.LazyFrame(df).select((10 - pql.col("x")).alias("result")).collect(),
        pl.LazyFrame(df).select((10 - pl.col("x")).alias("result")).collect(),
    )


def test_rfloordiv() -> None:
    df = pl.DataFrame({"x": [2, 3, 4]})
    assert_eq(
        pql.LazyFrame(df).select((10 // pql.col("x")).alias("result")).collect(),
        pl.LazyFrame(df).select((10 // pl.col("x")).alias("result")).collect(),
    )


def test_rmod() -> None:
    df = pl.DataFrame({"x": [3, 4, 5]})
    assert_eq(
        pql.LazyFrame(df).select((10 % pql.col("x")).alias("result")).collect(),
        pl.LazyFrame(df).select((10 % pl.col("x")).alias("result")).collect(),
    )


def test_rpow() -> None:
    df = pl.DataFrame({"x": [2, 3, 4]})
    assert_eq(
        pql.LazyFrame(df).select((2 ** pql.col("x")).alias("result")).collect(),
        pl.LazyFrame(df).select((2 ** pl.col("x")).alias("result")).collect(),
    )


def test_neg() -> None:
    df = pl.DataFrame({"x": [1, -2, 3]})
    assert_eq(
        pql.LazyFrame(df).select((-pql.col("x")).alias("result")).collect(),
        pl.LazyFrame(df).select((-pl.col("x")).alias("result")).collect(),
    )


def test_ne() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    assert_eq(
        pql.LazyFrame(df).filter(pql.col("x") != 2).collect(),
        pl.LazyFrame(df).filter(pl.col("x") != 2).collect(),
    )


def test_le() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    assert_eq(
        pql.LazyFrame(df).filter(pql.col("x") <= 2).collect(),
        pl.LazyFrame(df).filter(pl.col("x") <= 2).collect(),
    )


def test_sub() -> None:
    df = pl.DataFrame({"x": [10, 20, 30]})
    assert_eq(
        pql.LazyFrame(df).select((pql.col("x") - 5).alias("result")).collect(),
        pl.LazyFrame(df).select((pl.col("x") - 5).alias("result")).collect(),
    )


def test_floordiv() -> None:
    df = pl.DataFrame({"x": [10, 21, 32]})
    assert_eq(
        pql.LazyFrame(df).select((pql.col("x") // 3).alias("result")).collect(),
        pl.LazyFrame(df).select((pl.col("x") // 3).alias("result")).collect(),
    )


def test_hash_seed0() -> None:
    df = pl.DataFrame({"text": ["apple", "banana"]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("text").hash(seed=0).alias("h")).collect(),
        pl.LazyFrame(df).select(pl.col("text").hash(seed=0).alias("h")).collect(),
    )


def test_hash_seed42() -> None:
    df = pl.DataFrame({"text": ["apple", "banana"]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("text").hash(seed=42).alias("h")).collect(),
        pl.LazyFrame(df).select(pl.col("text").hash(seed=42).alias("h")).collect(),
    )


def test_lit() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    assert_eq(
        pql.LazyFrame(df).select(pql.lit(42).alias("constant")).collect(),
        pl.LazyFrame(df).select(pl.lit(42).alias("constant")).collect(),
    )


def test_all() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert_eq(
        pql.LazyFrame(df).select(pql.all()).collect(),
        pl.LazyFrame(df).select(pl.all()).collect(),
    )


def test_repr() -> None:
    expr = pql.col("name")
    assert "Expr" in repr(expr)


def test_cast() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").cast(pql.String).alias("x_str"))
        .collect(),
        pl.LazyFrame(df).select(pl.col("x").cast(pl.String).alias("x_str")).collect(),
    )


def test_and() -> None:
    df = pl.DataFrame(
        {"a": [True, True, False, False], "b": [True, False, True, False]}
    )
    assert_eq(
        pql.LazyFrame(df)
        .select((pql.col("a") & pql.col("b")).alias("result"))
        .collect(),
        pl.LazyFrame(df).select((pl.col("a") & pl.col("b")).alias("result")).collect(),
    )


def test_or() -> None:
    df = pl.DataFrame(
        {"a": [True, True, False, False], "b": [True, False, True, False]}
    )
    assert_eq(
        pql.LazyFrame(df)
        .select((pql.col("a") | pql.col("b")).alias("result"))
        .collect(),
        pl.LazyFrame(df).select((pl.col("a") | pl.col("b")).alias("result")).collect(),
    )


def test_not() -> None:
    df = pl.DataFrame({"a": [True, False, True]})
    assert_eq(
        pql.LazyFrame(df).select((~pql.col("a")).alias("result")).collect(),
        pl.LazyFrame(df).select((~pl.col("a")).alias("result")).collect(),
    )


def test_radd() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    assert_eq(
        pql.LazyFrame(df).select((10 + pql.col("x")).alias("result")).collect(),
        pl.LazyFrame(df).select((10 + pl.col("x")).alias("result")).collect(),
    )


def test_rmul() -> None:
    df = pl.DataFrame({"x": [2, 3, 4]})
    assert_eq(
        pql.LazyFrame(df).select((10 * pql.col("x")).alias("result")).collect(),
        pl.LazyFrame(df).select((10 * pl.col("x")).alias("result")).collect(),
    )


def test_rtruediv() -> None:
    df = pl.DataFrame({"x": [2.0, 4.0, 5.0]})
    assert_eq(
        pql.LazyFrame(df).select((10 / pql.col("x")).alias("result")).collect(),
        pl.LazyFrame(df).select((10 / pl.col("x")).alias("result")).collect(),
    )


def test_eq() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    assert_eq(
        pql.LazyFrame(df).filter(pql.col("x") == 2).collect(),
        pl.LazyFrame(df).filter(pl.col("x") == 2).collect(),
    )


def test_lt() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    assert_eq(
        pql.LazyFrame(df).filter(pql.col("x") < 3).collect(),
        pl.LazyFrame(df).filter(pl.col("x") < 3).collect(),
    )


def test_gt() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    assert_eq(
        pql.LazyFrame(df).filter(pql.col("x") > 2).collect(),
        pl.LazyFrame(df).filter(pl.col("x") > 2).collect(),
    )


def test_ge() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    assert_eq(
        pql.LazyFrame(df).filter(pql.col("x") >= 3).collect(),
        pl.LazyFrame(df).filter(pl.col("x") >= 3).collect(),
    )


def test_rand() -> None:
    df = pl.DataFrame({"a": [True, False], "b": [True, True]})
    assert_eq(
        pql.LazyFrame(df).select((True & pql.col("a")).alias("result")).collect(),
        pl.LazyFrame(df).select((True & pl.col("a")).alias("result")).collect(),
    )


def test_ror() -> None:
    df = pl.DataFrame({"a": [False, False], "b": [True, False]})
    assert_eq(
        pql.LazyFrame(df).select((False | pql.col("a")).alias("result")).collect(),
        pl.LazyFrame(df).select((False | pl.col("a")).alias("result")).collect(),
    )


def test_clip_lower_only() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").clip(lower_bound=2).alias("clipped"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("x").clip(lower_bound=2).alias("clipped"))
        .collect(),
    )


def test_clip_upper_only() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").clip(upper_bound=4).alias("clipped"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("x").clip(upper_bound=4).alias("clipped"))
        .collect(),
    )


def test_sort_by() -> None:
    df = pl.DataFrame({"name": ["Charlie", "Alice", "Bob"], "age": [35, 25, 30]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("name").sort_by("age")).collect(),
        pl.LazyFrame(df).select(pl.col("name").sort_by("age")).collect(),
    )


def test_head_expr() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").head(3)).collect(),
        pl.LazyFrame(df).select(pl.col("x").head(3)).collect(),
    )


def test_tail_expr() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").tail(3)).collect(),
        pl.LazyFrame(df).select(pl.col("x").tail(3)).collect(),
    )


def test_arg_sort() -> None:
    df = pl.DataFrame({"x": [3, 1, 2]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").arg_sort().alias("idx")).collect(),
        pl.LazyFrame(df).select(pl.col("x").arg_sort().alias("idx")).collect(),
    )


def test_arg_unique() -> None:
    df = pl.DataFrame({"x": [3, 1, 2, 3, 2, 1]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").arg_unique().alias("idx")).collect(),
        pl.LazyFrame(df).select(pl.col("x").arg_unique().alias("idx")).collect(),
    )


def test_arg_sort_descending() -> None:
    df = pl.DataFrame({"x": [3, 1, 2]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").arg_sort(descending=True).alias("idx"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("x").arg_sort(descending=True).alias("idx"))
        .collect(),
    )


def test_ewm_mean() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").ewm_mean().alias("ewm")).collect(),
        pl.LazyFrame(df).select(pl.col("x").ewm_mean().alias("ewm")).collect(),
    )


def test_ewm_mean_alpha() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").ewm_mean(alpha=0.5).alias("ewm"))
        .collect(),
        pl.LazyFrame(df).select(pl.col("x").ewm_mean(alpha=0.5).alias("ewm")).collect(),
    )


def test_ewm_mean_com() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").ewm_mean(com=0.5).alias("ewm")).collect(),
        pl.LazyFrame(df).select(pl.col("x").ewm_mean(com=0.5).alias("ewm")).collect(),
    )


def test_ewm_mean_span() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").ewm_mean(span=3).alias("ewm")).collect(),
        pl.LazyFrame(df).select(pl.col("x").ewm_mean(span=3).alias("ewm")).collect(),
    )


def test_ewm_mean_halflife() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").ewm_mean(half_life=2).alias("ewm"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("x").ewm_mean(half_life=2).alias("ewm"))
        .collect(),
    )


def test_peak_max() -> None:
    df = pl.DataFrame({"x": [1, 3, 2, 4, 2]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").peak_max()).collect(),
        pl.LazyFrame(df).select(pl.col("x").peak_max()).collect(),
    )


def test_peak_min() -> None:
    df = pl.DataFrame({"x": [3, 1, 2, 1, 3]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").peak_min()).collect(),
        pl.LazyFrame(df).select(pl.col("x").peak_min()).collect(),
    )


def test_hash() -> None:
    expr1 = pql.col("x")
    expr2 = pql.col("x")
    assert hash(expr1) == hash(expr2)


def test_repeat_by_int() -> None:
    df = pl.DataFrame({"x": ["a", "b", "c"]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").repeat_by(2).alias("repeated")).collect(),
        pl.LazyFrame(df).select(pl.col("x").repeat_by(2).alias("repeated")).collect(),
    )


def test_repeat_by_expr() -> None:
    df = pl.DataFrame({"x": ["a", "b", "c"], "n": [1, 2, 3]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").repeat_by(pql.col("n")).alias("repeated"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("x").repeat_by(pl.col("n")).alias("repeated"))
        .collect(),
    )


def test_arg_min() -> None:
    df = pl.DataFrame({"x": [3, 1, 2], "y": [10, 20, 30]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").arg_min().alias("idx")).collect(),
        pl.LazyFrame(df).select(pl.col("x").arg_min().alias("idx")).collect(),
    )


def test_arg_max() -> None:
    df = pl.DataFrame({"x": [3, 1, 2], "y": [10, 20, 30]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").arg_max().alias("idx")).collect(),
        pl.LazyFrame(df).select(pl.col("x").arg_max().alias("idx")).collect(),
    )


def test_mul() -> None:
    df = pl.DataFrame({"x": [2, 3, 4]})
    assert_eq(
        pql.LazyFrame(df).select((pql.col("x") * 5).alias("result")).collect(),
        pl.LazyFrame(df).select((pl.col("x") * 5).alias("result")).collect(),
    )


def test_truediv() -> None:
    df = pl.DataFrame({"x": [10.0, 20.0, 30.0]})
    assert_eq(
        pql.LazyFrame(df).select((pql.col("x") / 5).alias("result")).collect(),
        pl.LazyFrame(df).select((pl.col("x") / 5).alias("result")).collect(),
    )


def test_append() -> None:
    df = pl.DataFrame({"a": [[1, 2]], "b": [[3, 4]]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("a").append(pql.col("b")).alias("combined"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("a").append(pl.col("b")).alias("combined"))
        .collect(),
    )


def test_replace() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 2]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("x").replace(2, 99).alias("rep")).collect(),
        pl.LazyFrame(df).select(pl.col("x").replace(2, 99).alias("rep")).collect(),
    )


def test_shift_with_expr_fill_value() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    assert_eq(
        pql.LazyFrame(df).shift(1, fill_value=pql.lit(999)).collect(),
        pl.LazyFrame(df).shift(1, fill_value=pl.lit(999)).collect(),
    )


def test_map_elements() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").map_elements(lambda x: x + 1).alias("mapped"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("x").map_elements(lambda x: x + 1).alias("mapped"))
        .collect(),
    )


def test_map_batches() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("x").map_batches(lambda x: x * 2).alias("mapped"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("x").map_batches(lambda x: x * 2).alias("mapped"))
        .collect(),
    )
