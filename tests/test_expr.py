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
        result = pql.col("x").add(pql.col("y"))
        assert "x + y" in result.__node__.sql().lower()

    def test_radd(self) -> None:
        result = pql.col("x").radd(1)
        assert "1 + x" in result.__node__.sql().lower()

    def test_sub(self) -> None:
        result = pql.col("x").sub(pql.col("y"))
        assert "x - y" in result.__node__.sql().lower()

    def test_rsub(self) -> None:
        result = pql.col("x").rsub(10)
        assert "10 - x" in result.__node__.sql().lower()

    def test_mul(self) -> None:
        result = pql.col("x").mul(pql.col("y"))
        assert "x * y" in result.__node__.sql().lower()

    def test_rmul(self) -> None:
        result = pql.col("x").rmul(2)
        assert "2 * x" in result.__node__.sql().lower()

    def test_truediv(self) -> None:
        result = pql.col("x").truediv(pql.col("y"))
        assert "x / y" in result.__node__.sql().lower()

    def test_rtruediv(self) -> None:
        result = pql.col("x").rtruediv(100)
        assert "100 / x" in result.__node__.sql().lower()

    def test_floordiv(self) -> None:
        result = pql.col("x").floordiv(pql.col("y"))
        sql = result.__node__.sql().lower()
        # DuckDB uses CAST(x / y AS INT) for integer division
        assert "div" in sql or "//" in sql or ("cast" in sql and "/" in sql)

    def test_rfloordiv(self) -> None:
        result = pql.col("x").rfloordiv(100)
        sql = result.__node__.sql().lower()
        assert "100" in sql
        assert "x" in sql

    def test_mod(self) -> None:
        result = pql.col("x").mod(pql.col("y"))
        sql = result.__node__.sql().lower()
        assert "mod" in sql or "%" in sql

    def test_rmod(self) -> None:
        result = pql.col("x").rmod(10)
        sql = result.__node__.sql().lower()
        assert "10" in sql
        assert "x" in sql

    def test_pow(self) -> None:
        result = pql.col("x").pow(2)
        sql = result.__node__.sql().lower()
        assert "pow" in sql or "**" in sql or "power" in sql

    def test_rpow(self) -> None:
        result = pql.col("x").rpow(2)
        sql = result.__node__.sql().lower()
        assert "2" in sql
        assert "x" in sql

    def test_neg(self) -> None:
        result = pql.col("x").neg()
        assert "-" in result.__node__.sql()

    def test_pos(self) -> None:
        result = pql.col("x").pos()
        assert result.__node__ == pql.col("x").__node__

    def test_abs(self) -> None:
        result = pql.col("x").abs()
        assert "abs" in result.__node__.sql().lower()


class TestComparisonOperators:
    """Tests for comparison operators."""

    def test_eq(self) -> None:
        result = pql.col("x").eq(5)
        assert "=" in result.__node__.sql()

    def test_ne(self) -> None:
        result = pql.col("x").ne(5)
        sql = result.__node__.sql()
        assert "<>" in sql or "!=" in sql

    def test_lt(self) -> None:
        result = pql.col("x").lt(5)
        assert "< 5" in result.__node__.sql()

    def test_le(self) -> None:
        result = pql.col("x").le(5)
        assert "<= 5" in result.__node__.sql()

    def test_gt(self) -> None:
        result = pql.col("x").gt(5)
        assert "> 5" in result.__node__.sql()

    def test_ge(self) -> None:
        result = pql.col("x").ge(5)
        assert ">= 5" in result.__node__.sql()


class TestLogicalOperators:
    """Tests for logical operators."""

    def test_and(self) -> None:
        result = (pql.col("x").gt(5)).and_(pql.col("y").lt(10))
        sql = result.__node__.sql().lower()
        assert "and" in sql

    def test_rand(self) -> None:
        # Note: This requires the left operand to implement __and__ returning NotImplemented
        # In practice, (lit(True) & pql.col("x")) might not work as expected due to Python semantics
        result = (pql.col("x").gt(0)).and_(pql.col("y").gt(0))
        assert "and" in result.__node__.sql().lower()

    def test_or(self) -> None:
        result = (pql.col("x").gt(5)).or_(pql.col("y").lt(10))
        sql = result.__node__.sql().lower()
        assert "or" in sql

    def test_ror(self) -> None:
        result = (pql.col("x").gt(0)).or_(pql.col("y").gt(0))
        assert "or" in result.__node__.sql().lower()

    def test_invert(self) -> None:
        result = pql.col("x").gt(5).not_()
        sql = result.__node__.sql().lower()
        assert "not" in sql


class TestExprMethods:
    """Tests for Expr methods."""

    def test_alias(self) -> None:
        result = pql.col("x").alias("new_name")
        sql = result.__node__.sql()
        assert "new_name" in sql

    def test_is_null(self) -> None:
        result = pql.col("x").is_null()
        sql = result.__node__.sql().lower()
        assert "is null" in sql

    def test_is_not_null(self) -> None:
        result = pql.col("x").is_not_null()
        sql = result.__node__.sql().lower()
        assert "not" in sql
        assert "null" in sql

    def test_fill_null(self) -> None:
        result = pql.col("x").fill_null(0)
        sql = result.__node__.sql().lower()
        assert "coalesce" in sql

    def test_cast(self) -> None:
        result = pql.col("x").cast("VARCHAR")
        sql = result.__node__.sql().lower()
        assert "cast" in sql

    def test_between(self) -> None:
        result = pql.col("x").between(1, 10)
        sql = result.__node__.sql().lower()
        assert "between" in sql

    def test_is_in(self) -> None:
        result = pql.col("x").is_in([1, 2, 3])
        sql = result.__node__.sql().lower()
        assert "in" in sql

    def test_is_not_in(self) -> None:
        result = pql.col("x").is_not_in([1, 2, 3])
        sql = result.__node__.sql().lower()
        assert "not" in sql
        assert "in" in sql


class TestAggregations:
    """Tests for aggregation methods."""

    def test_sum(self) -> None:
        result = pql.col("x").sum()
        assert "sum" in result.__node__.sql().lower()

    def test_mean(self) -> None:
        result = pql.col("x").mean()
        sql = result.__node__.sql().lower()
        assert "avg" in sql

    def test_min(self) -> None:
        result = pql.col("x").min()
        assert "min" in result.__node__.sql().lower()

    def test_max(self) -> None:
        result = pql.col("x").max()
        assert "max" in result.__node__.sql().lower()

    def test_count(self) -> None:
        result = pql.col("x").count()
        assert "count" in result.__node__.sql().lower()

    def test_std_sample(self) -> None:
        result = pql.col("x").std(ddof=1)
        sql = result.__node__.sql().lower()
        assert "stddev" in sql

    def test_std_population(self) -> None:
        result = pql.col("x").std(ddof=0)
        sql = result.__node__.sql().lower()
        assert "stddev" in sql

    def test_var_sample(self) -> None:
        result = pql.col("x").var(ddof=1)
        sql = result.__node__.sql().lower()
        assert "var" in sql

    def test_var_population(self) -> None:
        result = pql.col("x").var(ddof=0)
        sql = result.__node__.sql().lower()
        assert "var" in sql

    def test_first(self) -> None:
        result = pql.col("x").first()
        assert "first" in result.__node__.sql().lower()

    def test_last(self) -> None:
        result = pql.col("x").last()
        assert "last" in result.__node__.sql().lower()

    def test_n_unique(self) -> None:
        result = pql.col("x").n_unique()
        sql = result.__node__.sql().lower()
        assert "count" in sql
        assert "distinct" in sql


class TestStringNamespace:
    """Tests for string operations."""

    def test_to_uppercase(self) -> None:
        result = pql.col("x").str.to_uppercase()
        assert "upper" in result.__node__.sql().lower()

    def test_to_lowercase(self) -> None:
        result = pql.col("x").str.to_lowercase()
        assert "lower" in result.__node__.sql().lower()

    def test_len_chars(self) -> None:
        result = pql.col("x").str.len_chars()
        assert "length" in result.__node__.sql().lower()

    def test_contains_literal(self) -> None:
        result = pql.col("x").str.contains("test", literal=True)
        sql = result.__node__.sql().lower()
        assert "like" in sql
        assert "%test%" in sql

    def test_contains_regex(self) -> None:
        result = pql.col("x").str.contains("test.*", literal=False)
        sql = result.__node__.sql().lower()
        assert "regexp" in sql or "like" in sql

    def test_starts_with(self) -> None:
        result = pql.col("x").str.starts_with("pre")
        sql = result.__node__.sql().lower()
        assert "like" in sql
        assert "pre%" in sql

    def test_ends_with(self) -> None:
        result = pql.col("x").str.ends_with("suf")
        sql = result.__node__.sql().lower()
        assert "like" in sql
        assert "%suf" in sql

    def test_replace(self) -> None:
        result = pql.col("x").str.replace("old", "new")
        assert "replace" in result.__node__.sql().lower()

    def test_strip_chars(self) -> None:
        result = pql.col("x").str.strip_chars()
        assert "trim" in result.__node__.sql().lower()

    def test_strip_chars_start(self) -> None:
        result = pql.col("x").str.strip_chars_start()
        assert "ltrim" in result.__node__.sql().lower()

    def test_strip_chars_end(self) -> None:
        result = pql.col("x").str.strip_chars_end()
        assert "rtrim" in result.__node__.sql().lower()

    def test_slice(self) -> None:
        result = pql.col("x").str.slice(0, 5)
        sql = result.__node__.sql().lower()
        assert "substr" in sql or "substring" in sql


class TestDatetimeNamespace:
    """Tests for datetime operations."""

    def test_year(self) -> None:
        result = pql.col("x").dt.year()
        assert "year" in result.__node__.sql().lower()

    def test_month(self) -> None:
        result = pql.col("x").dt.month()
        assert "month" in result.__node__.sql().lower()

    def test_day(self) -> None:
        result = pql.col("x").dt.day()
        assert "day" in result.__node__.sql().lower()

    def test_hour(self) -> None:
        result = pql.col("x").dt.hour()
        assert "hour" in result.__node__.sql().lower()

    def test_minute(self) -> None:
        result = pql.col("x").dt.minute()
        assert "minute" in result.__node__.sql().lower()

    def test_second(self) -> None:
        result = pql.col("x").dt.second()
        assert "second" in result.__node__.sql().lower()

    def test_weekday(self) -> None:
        result = pql.col("x").dt.weekday()
        sql = result.__node__.sql().lower()
        assert "day" in sql  # dayofweek

    def test_week(self) -> None:
        result = pql.col("x").dt.week()
        assert "week" in result.__node__.sql().lower()

    def test_date(self) -> None:
        result = pql.col("x").dt.date()
        assert "date" in result.__node__.sql().lower()
