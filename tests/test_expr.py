"""Tests for Expr class."""

from __future__ import annotations

import pql


class TestExprCreation:
    """Tests for creating expressions."""

    def test_col_creates_column_expr(self) -> None:
        expr = pql.col("x")
        assert "x" in repr(expr)

    def test_lit_creates_literal_expr(self) -> None:
        expr = pql.lit(42)
        assert "42" in repr(expr)

    def test_lit_string(self) -> None:
        expr = pql.lit("hello")
        assert "hello" in repr(expr)


class TestArithmeticOperators:
    """Tests for arithmetic operators."""

    def test_add(self) -> None:
        expr = pql.col("x").add(pql.col("y"))
        assert "x" in repr(expr)
        assert "y" in repr(expr)

    def test_add_literal(self) -> None:
        expr = pql.col("x").add(1)
        assert "x" in repr(expr)

    def test_sub(self) -> None:
        expr = pql.col("x").sub(pql.col("y"))
        assert "x" in repr(expr)

    def test_mul(self) -> None:
        expr = pql.col("x").mul(2)
        assert "x" in repr(expr)

    def test_truediv(self) -> None:
        expr = pql.col("x").truediv(2)
        assert "x" in repr(expr)

    def test_neg(self) -> None:
        expr = pql.col("x").neg()
        assert "x" in repr(expr)


class TestComparisonOperators:
    """Tests for comparison operators."""

    def test_eq(self) -> None:
        expr = pql.col("x").eq(5)
        assert "x" in repr(expr)

    def test_ne(self) -> None:
        expr = pql.col("x").ne(5)
        assert "x" in repr(expr)

    def test_lt(self) -> None:
        expr = pql.col("x").lt(5)
        assert "x" in repr(expr)

    def test_le(self) -> None:
        expr = pql.col("x").le(5)
        assert "x" in repr(expr)

    def test_gt(self) -> None:
        expr = pql.col("x").gt(5)
        assert "x" in repr(expr)

    def test_ge(self) -> None:
        expr = pql.col("x").ge(5)
        assert "x" in repr(expr)


class TestLogicalOperators:
    """Tests for logical operators."""

    def test_and(self) -> None:
        expr = pql.col("x").gt(5).and_(pql.col("y").lt(10))
        assert "x" in repr(expr)
        assert "y" in repr(expr)

    def test_or(self) -> None:
        expr = pql.col("x").gt(5).or_(pql.col("y").lt(10))
        assert "x" in repr(expr)

    def test_not(self) -> None:
        expr = pql.col("x").gt(5).not_()
        assert "x" in repr(expr)


class TestExprMethods:
    """Tests for Expr methods."""

    def test_alias(self) -> None:
        expr = pql.col("x").alias("new_name")
        assert "x" in repr(expr)  # alias may not show in repr

    def test_is_null(self) -> None:
        expr = pql.col("x").is_null()
        assert "x" in repr(expr)

    def test_is_not_null(self) -> None:
        expr = pql.col("x").is_not_null()
        assert "x" in repr(expr)

    def test_fill_null(self) -> None:
        expr = pql.col("x").fill_null(0)
        assert "x" in repr(expr)

    def test_between(self) -> None:
        expr = pql.col("x").between(1, 10)
        assert "x" in repr(expr)

    def test_is_in(self) -> None:
        expr = pql.col("x").is_in([1, 2, 3])
        assert "x" in repr(expr)

    def test_is_not_in(self) -> None:
        expr = pql.col("x").is_not_in([1, 2, 3])
        assert "x" in repr(expr)


class TestAggregations:
    """Tests for aggregation methods."""

    def test_sum(self) -> None:
        expr = pql.col("x").sum()
        assert "sum" in repr(expr).lower()

    def test_mean(self) -> None:
        expr = pql.col("x").mean()
        assert "avg" in repr(expr).lower()

    def test_min(self) -> None:
        expr = pql.col("x").min()
        assert "min" in repr(expr).lower()

    def test_max(self) -> None:
        expr = pql.col("x").max()
        assert "max" in repr(expr).lower()

    def test_count(self) -> None:
        expr = pql.col("x").count()
        assert "count" in repr(expr).lower()

    def test_first(self) -> None:
        expr = pql.col("x").first()
        assert "first" in repr(expr).lower()

    def test_last(self) -> None:
        expr = pql.col("x").last()
        assert "last" in repr(expr).lower()


class TestStringNamespace:
    """Tests for string operations."""

    def test_to_uppercase(self) -> None:
        expr = pql.col("x").str.to_uppercase()
        assert "upper" in repr(expr).lower()

    def test_to_lowercase(self) -> None:
        expr = pql.col("x").str.to_lowercase()
        assert "lower" in repr(expr).lower()

    def test_len_chars(self) -> None:
        expr = pql.col("x").str.len_chars()
        assert "length" in repr(expr).lower()

    def test_contains(self) -> None:
        expr = pql.col("x").str.contains("test")
        assert "x" in repr(expr)

    def test_starts_with(self) -> None:
        expr = pql.col("x").str.starts_with("pre")
        assert "x" in repr(expr)

    def test_ends_with(self) -> None:
        expr = pql.col("x").str.ends_with("suf")
        assert "x" in repr(expr)

    def test_replace(self) -> None:
        expr = pql.col("x").str.replace("old", "new")
        assert "replace" in repr(expr).lower()

    def test_strip_chars(self) -> None:
        expr = pql.col("x").str.strip_chars()
        assert "trim" in repr(expr).lower()


class TestDatetimeNamespace:
    """Tests for datetime operations."""

    def test_year(self) -> None:
        expr = pql.col("x").dt.year()
        assert "year" in repr(expr).lower()

    def test_month(self) -> None:
        expr = pql.col("x").dt.month()
        assert "month" in repr(expr).lower()

    def test_day(self) -> None:
        expr = pql.col("x").dt.day()
        assert "day" in repr(expr).lower()

    def test_hour(self) -> None:
        expr = pql.col("x").dt.hour()
        assert "hour" in repr(expr).lower()
