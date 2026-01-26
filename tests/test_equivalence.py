"""Tests for PQL/DuckDB and Polars equivalence on concrete data.

These tests verify that queries built with PQL produce the same results
as the equivalent Polars operations when executed on the same data.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pql


# ==================== Fixtures ====================
@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
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
    """Create a sample DataFrame with null values."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10.0, None, 30.0, None, 50.0],
            "category": ["A", "B", None, "A", "B"],
        }
    )


@pytest.fixture
def sample_df_strings() -> pl.DataFrame:
    """Create a sample DataFrame for string operations."""
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
    """Create a sample DataFrame for datetime operations."""
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
    """Create an orders DataFrame for aggregation tests."""
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
    """Create a left DataFrame for join tests."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "left_value": ["a", "b", "c", "d"],
        }
    )


@pytest.fixture
def right_df() -> pl.DataFrame:
    """Create a right DataFrame for join tests."""
    return pl.DataFrame(
        {
            "id": [2, 3, 4, 5],
            "right_value": ["x", "y", "z", "w"],
        }
    )


def _assert_frames_equal(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    *,
    check_row_order: bool = True,
    check_column_order: bool = True,
) -> None:
    """Assert that two DataFrames are equal."""
    if not check_column_order:
        actual = actual.select(sorted(actual.columns))
        expected = expected.select(sorted(expected.columns))
    if not check_row_order:
        actual = actual.sort(actual.columns)
        expected = expected.sort(expected.columns)
    assert_frame_equal(actual, expected, check_exact=False)


# ==================== Join Tests ====================
class TestJoinEquivalence:
    """Tests for join operation equivalence."""

    def test_inner_join(self, left_df: pl.DataFrame, right_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(left_df)
            .join(
                pql.LazyFrame(right_df),
                on="id",
                how="inner",
            )
            .collect()
        )
        polars_result = (
            left_df.lazy().join(right_df.lazy(), on="id", how="inner").collect()
        )
        _assert_frames_equal(pql_result, polars_result, check_row_order=False)

    def test_left_join(self, left_df: pl.DataFrame, right_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(left_df)
            .join(
                pql.LazyFrame(right_df),
                on="id",
                how="left",
            )
            .collect()
        )
        polars_result = (
            left_df.lazy().join(right_df.lazy(), on="id", how="left").collect()
        )
        _assert_frames_equal(pql_result, polars_result, check_row_order=False)


# ==================== Select Tests ====================
class TestSelectEquivalence:
    """Tests for select operation equivalence."""

    def test_select_single_column(self, sample_df: pl.DataFrame) -> None:
        pql_result = pql.LazyFrame(sample_df).select("name").collect()
        polars_result = sample_df.lazy().select("name").collect()
        _assert_frames_equal(pql_result, polars_result)

    def test_select_multiple_columns(self, sample_df: pl.DataFrame) -> None:
        pql_result = pql.LazyFrame(sample_df).select("name", "age", "salary").collect()

        polars_result = sample_df.lazy().select("name", "age", "salary").collect()
        _assert_frames_equal(pql_result, polars_result)

    def test_select_with_expression(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df)
            .select(
                pql.col("name"), pql.col("salary").mul(1.1).alias("salary_increase")
            )
            .collect()
        )
        polars_result = (
            sample_df.lazy()
            .select(
                pl.col("name"),
                pl.col("salary").mul(1.1).alias("salary_increase"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)


# ==================== Filter Tests ====================
class TestFilterEquivalence:
    """Tests for filter operation equivalence."""

    def test_filter_gt(self, sample_df: pl.DataFrame) -> None:
        pql_result = pql.LazyFrame(sample_df).filter(pql.col("age").gt(25)).collect()
        polars_result = sample_df.lazy().filter(pl.col("age").gt(25)).collect()
        _assert_frames_equal(pql_result, polars_result)

    def test_filter_eq_string(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df)
            .filter(pql.col("department").eq("Engineering"))
            .collect()
        )
        polars_result = (
            sample_df.lazy().filter(pl.col("department").eq("Engineering")).collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_filter_combined_and(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df)
            .filter(pql.col("age").gt(25).and_(pql.col("salary").ge(55000)))
            .collect()
        )
        polars_result = (
            sample_df.lazy()
            .filter((pl.col("age") > 25) & (pl.col("salary") >= 55000))
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_filter_combined_or(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df)
            .filter((pql.col("age").lt(25)).or_(pql.col("salary").gt(70000)))
            .collect()
        )
        polars_result = (
            sample_df.lazy()
            .filter((pl.col("age") < 25) | (pl.col("salary") > 70000))
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_filter_not(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df).filter(pql.col("is_active").not_()).collect()
        )
        polars_result = sample_df.lazy().filter(pl.col("is_active").not_()).collect()
        _assert_frames_equal(pql_result, polars_result)

    def test_filter_is_in(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df)
            .filter(pql.col("name").is_in(["Alice", "Bob", "Charlie"]))
            .collect()
        )
        polars_result = (
            sample_df.lazy()
            .filter(pl.col("name").is_in(["Alice", "Bob", "Charlie"]))
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_filter_between(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df).filter(pql.col("age").between(25, 30)).collect()
        )
        polars_result = (
            sample_df.lazy().filter(pl.col("age").is_between(25, 30)).collect()
        )
        _assert_frames_equal(pql_result, polars_result)


# ==================== Null Handling Tests ====================
class TestNullHandlingEquivalence:
    """Tests for null handling equivalence."""

    def test_filter_is_null(self, sample_df_with_nulls: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_with_nulls)
            .filter(pql.col("value").is_null())
            .collect()
        )
        polars_result = (
            sample_df_with_nulls.lazy().filter(pl.col("value").is_null()).collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_filter_is_not_null(self, sample_df_with_nulls: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_with_nulls)
            .filter(pql.col("value").is_not_null())
            .collect()
        )
        polars_result = (
            sample_df_with_nulls.lazy().filter(pl.col("value").is_not_null()).collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_fill_null(self, sample_df_with_nulls: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_with_nulls)
            .select(
                pql.col("id"),
                pql.col("value").fill_null(0.0).alias("value"),
            )
            .collect()
        )
        polars_result = (
            sample_df_with_nulls.lazy()
            .select(
                pl.col("id"),
                pl.col("value").fill_null(0.0),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)


# ==================== Aggregation Tests ====================
class TestAggregationEquivalence:
    """Tests for aggregation equivalence."""

    def test_group_by_sum(self, orders_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(orders_df)
            .group_by("customer_id")
            .agg(pql.col("amount").sum().alias("total_amount"))
            .collect()
        )
        polars_result = (
            orders_df.lazy()
            .group_by("customer_id")
            .agg(pl.col("amount").sum().alias("total_amount"))
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result, check_row_order=False)

    def test_group_by_mean(self, orders_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(orders_df)
            .group_by("customer_id")
            .agg(pql.col("amount").mean().alias("avg_amount"))
            .collect()
        )
        polars_result = (
            orders_df.lazy()
            .group_by("customer_id")
            .agg(pl.col("amount").mean().alias("avg_amount"))
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result, check_row_order=False)

    def test_group_by_count(self, orders_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(orders_df)
            .group_by("customer_id")
            .agg(pql.col("order_id").count().alias("order_count"))
            .collect()
        )
        polars_result = (
            orders_df.lazy()
            .group_by("customer_id")
            .agg(pl.col("order_id").count().alias("order_count"))
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result, check_row_order=False)

    def test_group_by_min_max(self, orders_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(orders_df)
            .group_by("customer_id")
            .agg(
                pql.col("amount").min().alias("min_amount"),
                pql.col("amount").max().alias("max_amount"),
            )
            .collect()
        )
        polars_result = (
            orders_df.lazy()
            .group_by("customer_id")
            .agg(
                pl.col("amount").min().alias("min_amount"),
                pl.col("amount").max().alias("max_amount"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result, check_row_order=False)

    def test_group_by_multiple_columns(self, orders_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(orders_df)
            .group_by("customer_id", "status")
            .agg(pql.col("amount").sum().alias("total_amount"))
            .collect()
        )
        polars_result = (
            orders_df.lazy()
            .group_by("customer_id", "status")
            .agg(pl.col("amount").sum().alias("total_amount"))
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result, check_row_order=False)


# ==================== Sort Tests ====================
class TestSortEquivalence:
    """Tests for sort operation equivalence."""

    def test_sort_ascending(self, sample_df: pl.DataFrame) -> None:
        pql_result = pql.LazyFrame(sample_df).sort("age").collect()
        polars_result = sample_df.lazy().sort("age").collect()
        _assert_frames_equal(pql_result, polars_result)

    def test_sort_descending(self, sample_df: pl.DataFrame) -> None:
        pql_result = pql.LazyFrame(sample_df).sort("salary", descending=True).collect()
        polars_result = sample_df.lazy().sort("salary", descending=True).collect()
        _assert_frames_equal(pql_result, polars_result)

    def test_sort_multiple_columns(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df)
            .sort("department", "age", descending=[False, True])
            .collect()
        )
        polars_result = (
            sample_df.lazy()
            .sort("department", "age", descending=[False, True])
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)


# ==================== Limit Tests ====================
class TestLimitEquivalence:
    """Tests for limit operation equivalence."""

    def test_limit(self, sample_df: pl.DataFrame) -> None:
        pql_result = pql.LazyFrame(sample_df).sort("id").limit(3).collect()
        polars_result = sample_df.lazy().sort("id").limit(3).collect()
        _assert_frames_equal(pql_result, polars_result)

    def test_head(self, sample_df: pl.DataFrame) -> None:
        pql_result = pql.LazyFrame(sample_df).sort("id").head(2).collect()
        polars_result = sample_df.lazy().sort("id").head(2).collect()
        _assert_frames_equal(pql_result, polars_result)


# ==================== Arithmetic Tests ====================
class TestArithmeticEquivalence:
    """Tests for arithmetic operation equivalence."""

    def test_add(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df)
            .select(
                pql.col("id"),
                (pql.col("age").add(10)).alias("age_plus_10"),
            )
            .collect()
        )
        polars_result = (
            sample_df.lazy()
            .select(
                pl.col("id"),
                (pl.col("age") + 10).alias("age_plus_10"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_multiply(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df)
            .select(
                pql.col("id"),
                pql.col("salary").mul(2).alias("double_salary"),
            )
            .collect()
        )
        polars_result = (
            sample_df.lazy()
            .select(
                pl.col("id"),
                (pl.col("salary").mul(2)).alias("double_salary"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_divide(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df)
            .select(
                pql.col("id"),
                pql.col("salary").truediv(1000).alias("salary_k"),
            )
            .collect()
        )
        polars_result = (
            sample_df.lazy()
            .select(
                pl.col("id"),
                (pl.col("salary").truediv(1000)).alias("salary_k"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_mod(self, sample_df: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df)
            .select(
                pql.col("id"),
                pql.col("age").mod(10).alias("age_mod_10"),
            )
            .collect()
        )
        polars_result = (
            sample_df.lazy()
            .select(
                pl.col("id"),
                (pl.col("age").mod(10)).alias("age_mod_10"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_abs(self) -> None:
        abs_df = pl.DataFrame({"x": [-5, -3, 0, 3, 5]})
        pql_result = (
            pql.LazyFrame(abs_df)
            .select(
                pql.col("x"),
                pql.col("x").abs().alias("abs_x"),
            )
            .collect()
        )
        polars_result = (
            abs_df.lazy()
            .select(
                pl.col("x"),
                pl.col("x").abs().alias("abs_x"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)


# ==================== String Tests ====================
class TestStringEquivalence:
    """Tests for string operation equivalence."""

    def test_to_uppercase(self, sample_df_strings: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_strings)
            .select(
                pql.col("text"),
                pql.col("text").str.to_uppercase().alias("upper"),
            )
            .collect()
        )
        polars_result = (
            sample_df_strings.lazy()
            .select(
                pl.col("text"),
                pl.col("text").str.to_uppercase().alias("upper"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_to_lowercase(self, sample_df_strings: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_strings)
            .select(
                pql.col("text"),
                pql.col("text").str.to_lowercase().alias("lower"),
            )
            .collect()
        )
        polars_result = (
            sample_df_strings.lazy()
            .select(
                pl.col("text"),
                pl.col("text").str.to_lowercase().alias("lower"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_len_chars(self, sample_df_strings: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_strings)
            .select(
                pql.col("text"),
                pql.col("text").str.len_chars().alias("length"),
            )
            .collect()
        )
        polars_result = (
            sample_df_strings.lazy()
            .select(
                pl.col("text"),
                pl.col("text").str.len_chars().alias("length"),
            )
            .collect()
        )
        # Cast to match types (DuckDB returns bigint, Polars returns UInt32)
        pql_result = pql_result.cast({"length": pl.UInt32})
        _assert_frames_equal(pql_result, polars_result)

    def test_contains_literal(self, sample_df_strings: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_strings)
            .select(
                pql.col("text"),
                pql.col("text").str.contains("lo", literal=True).alias("contains_lo"),
            )
            .collect()
        )
        polars_result = (
            sample_df_strings.lazy()
            .select(
                pl.col("text"),
                pl.col("text").str.contains("lo", literal=True).alias("contains_lo"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_starts_with(self, sample_df_strings: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_strings)
            .select(
                pql.col("text"),
                pql.col("text").str.starts_with("Hello").alias("starts_hello"),
            )
            .collect()
        )
        polars_result = (
            sample_df_strings.lazy()
            .select(
                pl.col("text"),
                pl.col("text").str.starts_with("Hello").alias("starts_hello"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_ends_with(self, sample_df_strings: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_strings)
            .select(
                pql.col("text"),
                pql.col("text").str.ends_with("suffix").alias("ends_suffix"),
            )
            .collect()
        )
        polars_result = (
            sample_df_strings.lazy()
            .select(
                pl.col("text"),
                pl.col("text").str.ends_with("suffix").alias("ends_suffix"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_replace(self, sample_df_strings: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_strings)
            .select(
                pql.col("text"),
                pql.col("text").str.replace("Hello", "Hi").alias("replaced"),
            )
            .collect()
        )
        polars_result = (
            sample_df_strings.lazy()
            .select(
                pl.col("text"),
                pl.col("text")
                .str.replace("Hello", "Hi", literal=True)
                .alias("replaced"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_strip(self, sample_df_strings: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_strings)
            .select(
                pql.col("text"),
                pql.col("text").str.strip_chars().alias("stripped"),
            )
            .collect()
        )
        polars_result = (
            sample_df_strings.lazy()
            .select(
                pl.col("text"),
                pl.col("text").str.strip_chars().alias("stripped"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)


# ==================== Datetime Tests ====================
class TestDatetimeEquivalence:
    """Tests for datetime operation equivalence."""

    def test_year(self, sample_df_dates: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_dates)
            .select(
                pql.col("ts").dt.year().alias("year"),
            )
            .collect()
        )
        polars_result = (
            sample_df_dates.lazy()
            .select(
                pl.col("ts").dt.year().alias("year"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_month(self, sample_df_dates: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_dates)
            .select(
                pql.col("ts").dt.month().alias("month"),
            )
            .collect()
        )
        polars_result = (
            sample_df_dates.lazy()
            .select(
                pl.col("ts").dt.month().alias("month"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_day(self, sample_df_dates: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_dates)
            .select(
                pql.col("ts").dt.day().alias("day"),
            )
            .collect()
        )
        polars_result = (
            sample_df_dates.lazy()
            .select(
                pl.col("ts").dt.day().alias("day"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)

    def test_hour(self, sample_df_dates: pl.DataFrame) -> None:
        pql_result = (
            pql.LazyFrame(sample_df_dates)
            .select(
                pql.col("ts").dt.hour().alias("hour"),
            )
            .collect()
        )
        polars_result = (
            sample_df_dates.lazy()
            .select(
                pl.col("ts").dt.hour().alias("hour"),
            )
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)


# ==================== Complex Pipeline Tests ====================
class TestComplexPipelineEquivalence:
    """Tests for complex pipeline equivalence."""

    def test_filter_group_sort_limit(self, orders_df: pl.DataFrame) -> None:
        """Test a complete analytical pipeline."""
        pql_result = (
            pql.LazyFrame(orders_df)
            .filter(pql.col("status").eq("completed"))
            .group_by("customer_id")
            .agg(
                pql.col("amount").sum().alias("total_amount"),
                pql.col("order_id").count().alias("order_count"),
            )
            .sort("total_amount", descending=True)
            .limit(2)
            .collect()
        )
        polars_result = (
            orders_df.lazy()
            .filter(pl.col("status") == "completed")
            .group_by("customer_id")
            .agg(
                pl.col("amount").sum().alias("total_amount"),
                pl.col("order_id").count().alias("order_count"),
            )
            .sort("total_amount", descending=True)
            .limit(2)
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result, check_row_order=False)

    def test_with_columns_filter(self, sample_df: pl.DataFrame) -> None:
        """Test with_columns followed by filter."""
        pql_result = (
            pql.LazyFrame(sample_df)
            .select(
                pql.col("name"),
                pql.col("salary"),
                (pql.col("salary").mul(12)).alias("annual_salary"),
            )
            .filter(pql.col("annual_salary").gt(600000))
            .collect()
        )
        polars_result = (
            sample_df.lazy()
            .select(
                pl.col("name"),
                pl.col("salary"),
                (pl.col("salary").mul(12)).alias("annual_salary"),
            )
            .filter(pl.col("annual_salary").gt(600000))
            .collect()
        )
        _assert_frames_equal(pql_result, polars_result)


# ==================== Distinct Tests ====================
class TestDistinctEquivalence:
    """Tests for distinct operation equivalence."""

    def test_distinct(self) -> None:
        distinct_df = pl.DataFrame(
            {
                "a": [1, 1, 2, 2, 3],
                "b": ["x", "x", "y", "z", "z"],
            }
        )
        pql_result = pql.LazyFrame(distinct_df).distinct().sort("a", "b").collect()
        polars_result = distinct_df.lazy().unique().sort("a", "b").collect()
        _assert_frames_equal(pql_result, polars_result)
