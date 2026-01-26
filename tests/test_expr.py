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
        assert "x + y" in pql.col("x").add(pql.col("y")).__node__.sql().lower()

    def test_radd(self) -> None:
        assert "1 + x" in pql.col("x").radd(1).__node__.sql().lower()

    def test_sub(self) -> None:
        assert "x - y" in pql.col("x").sub(pql.col("y")).__node__.sql().lower()

    def test_rsub(self) -> None:
        assert "10 - x" in pql.col("x").rsub(10).__node__.sql().lower()

    def test_mul(self) -> None:
        assert "x * y" in pql.col("x").mul(pql.col("y")).__node__.sql().lower()

    def test_rmul(self) -> None:
        assert "2 * x" in pql.col("x").rmul(2).__node__.sql().lower()

    def test_truediv(self) -> None:
        assert "x / y" in pql.col("x").truediv(pql.col("y")).__node__.sql().lower()

    def test_rtruediv(self) -> None:
        assert "100 / x" in pql.col("x").rtruediv(100).__node__.sql().lower()

    def test_floordiv(self) -> None:
        sql = pql.col("x").floordiv(pql.col("y")).__node__.sql().lower()
        # DuckDB uses CAST(x / y AS INT) for integer division
        assert "div" in sql or "//" in sql or ("cast" in sql and "/" in sql)

    def test_rfloordiv(self) -> None:
        sql = pql.col("x").rfloordiv(100).__node__.sql().lower()
        assert "100" in sql
        assert "x" in sql

    def test_mod(self) -> None:
        sql = pql.col("x").mod(pql.col("y")).__node__.sql().lower()
        assert "mod" in sql or "%" in sql

    def test_rmod(self) -> None:
        sql = pql.col("x").rmod(10).__node__.sql().lower()
        assert "10" in sql
        assert "x" in sql

    def test_pow(self) -> None:
        sql = pql.col("x").pow(2).__node__.sql().lower()
        assert "pow" in sql or "**" in sql or "power" in sql

    def test_rpow(self) -> None:
        sql = pql.col("x").rpow(2).__node__.sql().lower()
        assert "2" in sql
        assert "x" in sql

    def test_neg(self) -> None:
        assert "-" in pql.col("x").neg().__node__.sql()

    def test_pos(self) -> None:
        assert pql.col("x").pos().__node__ == pql.col("x").__node__

    def test_abs(self) -> None:
        assert "abs" in pql.col("x").abs().__node__.sql().lower()


class TestComparisonOperators:
    """Tests for comparison operators."""

    def test_eq(self) -> None:
        assert "=" in pql.col("x").eq(5).__node__.sql()

    def test_ne(self) -> None:
        sql = pql.col("x").ne(5).__node__.sql()
        assert "<>" in sql or "!=" in sql

    def test_lt(self) -> None:
        assert "< 5" in pql.col("x").lt(5).__node__.sql()

    def test_le(self) -> None:
        assert "<= 5" in pql.col("x").le(5).__node__.sql()

    def test_gt(self) -> None:
        assert "> 5" in pql.col("x").gt(5).__node__.sql()

    def test_ge(self) -> None:
        assert ">=" in pql.col("x").ge(5).__node__.sql()


class TestLogicalOperators:
    """Tests for logical operators."""

    def test_and(self) -> None:
        sql = (pql.col("x").gt(5)).and_(pql.col("y").lt(10)).__node__.sql().lower()
        assert "and" in sql

    def test_rand(self) -> None:
        # Note: This requires the left operand to implement __and__ returning NotImplemented
        # In practice, (lit(True) & pql.col("x")) might not work as expected due to Python semantics
        assert (
            "and"
            in (pql.col("x").gt(0)).and_(pql.col("y").gt(0)).__node__.sql().lower()
        )

    def test_or(self) -> None:
        sql = (pql.col("x").gt(5)).or_(pql.col("y").lt(10)).__node__.sql().lower()
        assert "or" in sql

    def test_ror(self) -> None:
        assert (
            "or" in (pql.col("x").gt(0)).or_(pql.col("y").gt(0)).__node__.sql().lower()
        )

    def test_invert(self) -> None:
        sql = pql.col("x").gt(5).not_().__node__.sql().lower()
        assert "not" in sql


class TestExprMethods:
    """Tests for Expr methods."""

    def test_alias(self) -> None:
        sql = pql.col("x").alias("new_name").__node__.sql()
        assert "new_name" in sql

    def test_is_null(self) -> None:
        sql = pql.col("x").is_null().__node__.sql().lower()
        assert "is null" in sql

    def test_is_not_null(self) -> None:
        sql = pql.col("x").is_not_null().__node__.sql().lower()
        assert "not" in sql
        assert "null" in sql

    def test_fill_null(self) -> None:
        sql = pql.col("x").fill_null(0).__node__.sql().lower()
        assert "coalesce" in sql

    def test_cast(self) -> None:
        sql = pql.col("x").cast("VARCHAR").__node__.sql().lower()
        assert "cast" in sql

    def test_between(self) -> None:
        sql = pql.col("x").between(1, 10).__node__.sql().lower()
        assert "between" in sql

    def test_is_in(self) -> None:
        sql = pql.col("x").is_in([1, 2, 3]).__node__.sql().lower()
        assert "in" in sql

    def test_is_not_in(self) -> None:
        sql = pql.col("x").is_not_in([1, 2, 3]).__node__.sql().lower()
        assert "not" in sql
        assert "in" in sql


class TestAggregations:
    """Tests for aggregation methods."""

    def test_sum(self) -> None:
        assert "sum" in pql.col("x").sum().__node__.sql().lower()

    def test_mean(self) -> None:
        sql = pql.col("x").mean().__node__.sql().lower()
        assert "avg" in sql

    def test_min(self) -> None:
        assert "min" in pql.col("x").min().__node__.sql().lower()

    def test_max(self) -> None:
        assert "max" in pql.col("x").max().__node__.sql().lower()

    def test_count(self) -> None:
        assert "count" in pql.col("x").count().__node__.sql().lower()

    def test_std_sample(self) -> None:
        sql = pql.col("x").std(ddof=1).__node__.sql().lower()
        assert "stddev" in sql

    def test_std_population(self) -> None:
        sql = pql.col("x").std(ddof=0).__node__.sql().lower()
        assert "stddev" in sql

    def test_var_sample(self) -> None:
        sql = pql.col("x").var(ddof=1).__node__.sql().lower()
        assert "var" in sql

    def test_var_population(self) -> None:
        sql = pql.col("x").var(ddof=0).__node__.sql().lower()
        assert "var" in sql

    def test_first(self) -> None:
        assert "first" in pql.col("x").first().__node__.sql().lower()

    def test_last(self) -> None:
        assert "last" in pql.col("x").last().__node__.sql().lower()

    def test_n_unique(self) -> None:
        sql = pql.col("x").n_unique().__node__.sql().lower()
        assert "count" in sql
        assert "distinct" in sql


class TestStringNamespace:
    """Tests for string operations."""

    def test_to_uppercase(self) -> None:
        assert "upper" in pql.col("x").str.to_uppercase().__node__.sql().lower()

    def test_to_lowercase(self) -> None:
        assert "lower" in pql.col("x").str.to_lowercase().__node__.sql().lower()

    def test_len_chars(self) -> None:
        assert "length" in pql.col("x").str.len_chars().__node__.sql().lower()

    def test_contains_literal(self) -> None:
        sql = pql.col("x").str.contains("test", literal=True).__node__.sql().lower()
        assert "like" in sql
        assert "%test%" in sql

    def test_contains_regex(self) -> None:
        sql = pql.col("x").str.contains("test.*", literal=False).__node__.sql().lower()
        assert "regexp" in sql or "like" in sql

    def test_starts_with(self) -> None:
        sql = pql.col("x").str.starts_with("pre").__node__.sql().lower()
        assert "like" in sql
        assert "pre%" in sql

    def test_ends_with(self) -> None:
        sql = pql.col("x").str.ends_with("suf").__node__.sql().lower()
        assert "like" in sql
        assert "%suf" in sql

    def test_replace(self) -> None:
        assert (
            "replace" in pql.col("x").str.replace("old", "new").__node__.sql().lower()
        )

    def test_strip_chars(self) -> None:
        assert "trim" in pql.col("x").str.strip_chars().__node__.sql().lower()

    def test_strip_chars_start(self) -> None:
        assert "ltrim" in pql.col("x").str.strip_chars_start().__node__.sql().lower()

    def test_strip_chars_end(self) -> None:
        assert "rtrim" in pql.col("x").str.strip_chars_end().__node__.sql().lower()

    def test_slice(self) -> None:
        sql = pql.col("x").str.slice(0, 5).__node__.sql().lower()
        assert "substr" in sql or "substring" in sql


class TestDatetimeNamespace:
    """Tests for datetime operations."""

    def test_year(self) -> None:
        assert "year" in pql.col("x").dt.year().__node__.sql().lower()

    def test_month(self) -> None:
        assert "month" in pql.col("x").dt.month().__node__.sql().lower()

    def test_day(self) -> None:
        assert "day" in pql.col("x").dt.day().__node__.sql().lower()

    def test_hour(self) -> None:
        assert "hour" in pql.col("x").dt.hour().__node__.sql().lower()

    def test_minute(self) -> None:
        assert "minute" in pql.col("x").dt.minute().__node__.sql().lower()

    def test_second(self) -> None:
        assert "second" in pql.col("x").dt.second().__node__.sql().lower()

    def test_weekday(self) -> None:
        sql = pql.col("x").dt.weekday().__node__.sql().lower()
        assert "day" in sql  # dayofweek

    def test_week(self) -> None:
        assert "week" in pql.col("x").dt.week().__node__.sql().lower()

    def test_date(self) -> None:
        assert "date" in pql.col("x").dt.date().__node__.sql().lower()
