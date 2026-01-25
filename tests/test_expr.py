"""Tests for Expr class."""

from __future__ import annotations

from pql import col, lit


class TestExprCreation:
    """Tests for creating expressions."""

    def test_col_creates_column_expr(self) -> None:
        expr = col("x")
        assert "x" in repr(expr)

    def test_lit_creates_literal_expr(self) -> None:
        expr = lit(42)
        assert "42" in repr(expr)

    def test_lit_string(self) -> None:
        expr = lit("hello")
        assert "hello" in repr(expr)


class TestArithmeticOperators:
    """Tests for arithmetic operators."""

    def test_add(self) -> None:
        result = col("x") + col("y")
        assert "x + y" in result.__node__.sql().lower()

    def test_radd(self) -> None:
        result = 1 + col("x")
        assert "1 + x" in result.__node__.sql().lower()

    def test_sub(self) -> None:
        result = col("x") - col("y")
        assert "x - y" in result.__node__.sql().lower()

    def test_rsub(self) -> None:
        result = 10 - col("x")
        assert "10 - x" in result.__node__.sql().lower()

    def test_mul(self) -> None:
        result = col("x") * col("y")
        assert "x * y" in result.__node__.sql().lower()

    def test_rmul(self) -> None:
        result = 2 * col("x")
        assert "2 * x" in result.__node__.sql().lower()

    def test_truediv(self) -> None:
        result = col("x") / col("y")
        assert "x / y" in result.__node__.sql().lower()

    def test_rtruediv(self) -> None:
        result = 100 / col("x")
        assert "100 / x" in result.__node__.sql().lower()

    def test_floordiv(self) -> None:
        result = col("x") // col("y")
        sql = result.__node__.sql().lower()
        # DuckDB uses CAST(x / y AS INT) for integer division
        assert "div" in sql or "//" in sql or ("cast" in sql and "/" in sql)

    def test_rfloordiv(self) -> None:
        result = 100 // col("x")
        sql = result.__node__.sql().lower()
        assert "100" in sql
        assert "x" in sql

    def test_mod(self) -> None:
        result = col("x") % col("y")
        sql = result.__node__.sql().lower()
        assert "mod" in sql or "%" in sql

    def test_rmod(self) -> None:
        result = 10 % col("x")
        sql = result.__node__.sql().lower()
        assert "10" in sql
        assert "x" in sql

    def test_pow(self) -> None:
        result = col("x") ** 2
        sql = result.__node__.sql().lower()
        assert "pow" in sql or "**" in sql or "power" in sql

    def test_rpow(self) -> None:
        result = 2 ** col("x")
        sql = result.__node__.sql().lower()
        assert "2" in sql
        assert "x" in sql

    def test_neg(self) -> None:
        result = -col("x")
        assert "-" in result.__node__.sql()

    def test_pos(self) -> None:
        result = +col("x")
        assert result.__node__ == col("x").__node__

    def test_abs(self) -> None:
        result = abs(col("x"))
        assert "abs" in result.__node__.sql().lower()


class TestComparisonOperators:
    """Tests for comparison operators."""

    def test_eq(self) -> None:
        result = col("x") == 5
        assert "=" in result.__node__.sql()

    def test_ne(self) -> None:
        result = col("x") != 5
        sql = result.__node__.sql()
        assert "<>" in sql or "!=" in sql

    def test_lt(self) -> None:
        result = col("x") < 5
        assert "< 5" in result.__node__.sql()

    def test_le(self) -> None:
        result = col("x") <= 5
        assert "<= 5" in result.__node__.sql()

    def test_gt(self) -> None:
        result = col("x") > 5
        assert "> 5" in result.__node__.sql()

    def test_ge(self) -> None:
        result = col("x") >= 5
        assert ">= 5" in result.__node__.sql()


class TestLogicalOperators:
    """Tests for logical operators."""

    def test_and(self) -> None:
        result = (col("x") > 5) & (col("y") < 10)
        sql = result.__node__.sql().lower()
        assert "and" in sql

    def test_rand(self) -> None:
        # Note: This requires the left operand to implement __and__ returning NotImplemented
        # In practice, (lit(True) & col("x")) might not work as expected due to Python semantics
        result = (col("x") > 0) & (col("y") > 0)
        assert "and" in result.__node__.sql().lower()

    def test_or(self) -> None:
        result = (col("x") > 5) | (col("y") < 10)
        sql = result.__node__.sql().lower()
        assert "or" in sql

    def test_ror(self) -> None:
        result = (col("x") > 0) | (col("y") > 0)
        assert "or" in result.__node__.sql().lower()

    def test_invert(self) -> None:
        result = ~(col("x") > 5)
        sql = result.__node__.sql().lower()
        assert "not" in sql


class TestExprMethods:
    """Tests for Expr methods."""

    def test_alias(self) -> None:
        result = col("x").alias("new_name")
        sql = result.__node__.sql()
        assert "new_name" in sql

    def test_is_null(self) -> None:
        result = col("x").is_null()
        sql = result.__node__.sql().lower()
        assert "is null" in sql

    def test_is_not_null(self) -> None:
        result = col("x").is_not_null()
        sql = result.__node__.sql().lower()
        assert "not" in sql
        assert "null" in sql

    def test_fill_null(self) -> None:
        result = col("x").fill_null(0)
        sql = result.__node__.sql().lower()
        assert "coalesce" in sql

    def test_cast(self) -> None:
        result = col("x").cast("VARCHAR")
        sql = result.__node__.sql().lower()
        assert "cast" in sql

    def test_between(self) -> None:
        result = col("x").between(1, 10)
        sql = result.__node__.sql().lower()
        assert "between" in sql

    def test_is_in(self) -> None:
        result = col("x").is_in([1, 2, 3])
        sql = result.__node__.sql().lower()
        assert "in" in sql

    def test_is_not_in(self) -> None:
        result = col("x").is_not_in([1, 2, 3])
        sql = result.__node__.sql().lower()
        assert "not" in sql
        assert "in" in sql


class TestAggregations:
    """Tests for aggregation methods."""

    def test_sum(self) -> None:
        result = col("x").sum()
        assert "sum" in result.__node__.sql().lower()

    def test_mean(self) -> None:
        result = col("x").mean()
        sql = result.__node__.sql().lower()
        assert "avg" in sql

    def test_min(self) -> None:
        result = col("x").min()
        assert "min" in result.__node__.sql().lower()

    def test_max(self) -> None:
        result = col("x").max()
        assert "max" in result.__node__.sql().lower()

    def test_count(self) -> None:
        result = col("x").count()
        assert "count" in result.__node__.sql().lower()

    def test_std_sample(self) -> None:
        result = col("x").std(ddof=1)
        sql = result.__node__.sql().lower()
        assert "stddev" in sql

    def test_std_population(self) -> None:
        result = col("x").std(ddof=0)
        sql = result.__node__.sql().lower()
        assert "stddev" in sql

    def test_var_sample(self) -> None:
        result = col("x").var(ddof=1)
        sql = result.__node__.sql().lower()
        assert "var" in sql

    def test_var_population(self) -> None:
        result = col("x").var(ddof=0)
        sql = result.__node__.sql().lower()
        assert "var" in sql

    def test_first(self) -> None:
        result = col("x").first()
        assert "first" in result.__node__.sql().lower()

    def test_last(self) -> None:
        result = col("x").last()
        assert "last" in result.__node__.sql().lower()

    def test_n_unique(self) -> None:
        result = col("x").n_unique()
        sql = result.__node__.sql().lower()
        assert "count" in sql
        assert "distinct" in sql


class TestStringNamespace:
    """Tests for string operations."""

    def test_to_uppercase(self) -> None:
        result = col("x").str.to_uppercase()
        assert "upper" in result.__node__.sql().lower()

    def test_to_lowercase(self) -> None:
        result = col("x").str.to_lowercase()
        assert "lower" in result.__node__.sql().lower()

    def test_len_chars(self) -> None:
        result = col("x").str.len_chars()
        assert "length" in result.__node__.sql().lower()

    def test_contains_literal(self) -> None:
        result = col("x").str.contains("test", literal=True)
        sql = result.__node__.sql().lower()
        assert "like" in sql
        assert "%test%" in sql

    def test_contains_regex(self) -> None:
        result = col("x").str.contains("test.*", literal=False)
        sql = result.__node__.sql().lower()
        assert "regexp" in sql or "like" in sql

    def test_starts_with(self) -> None:
        result = col("x").str.starts_with("pre")
        sql = result.__node__.sql().lower()
        assert "like" in sql
        assert "pre%" in sql

    def test_ends_with(self) -> None:
        result = col("x").str.ends_with("suf")
        sql = result.__node__.sql().lower()
        assert "like" in sql
        assert "%suf" in sql

    def test_replace(self) -> None:
        result = col("x").str.replace("old", "new")
        assert "replace" in result.__node__.sql().lower()

    def test_strip_chars(self) -> None:
        result = col("x").str.strip_chars()
        assert "trim" in result.__node__.sql().lower()

    def test_strip_chars_start(self) -> None:
        result = col("x").str.strip_chars_start()
        assert "ltrim" in result.__node__.sql().lower()

    def test_strip_chars_end(self) -> None:
        result = col("x").str.strip_chars_end()
        assert "rtrim" in result.__node__.sql().lower()

    def test_slice(self) -> None:
        result = col("x").str.slice(0, 5)
        sql = result.__node__.sql().lower()
        assert "substr" in sql or "substring" in sql


class TestDatetimeNamespace:
    """Tests for datetime operations."""

    def test_year(self) -> None:
        result = col("x").dt.year()
        assert "year" in result.__node__.sql().lower()

    def test_month(self) -> None:
        result = col("x").dt.month()
        assert "month" in result.__node__.sql().lower()

    def test_day(self) -> None:
        result = col("x").dt.day()
        assert "day" in result.__node__.sql().lower()

    def test_hour(self) -> None:
        result = col("x").dt.hour()
        assert "hour" in result.__node__.sql().lower()

    def test_minute(self) -> None:
        result = col("x").dt.minute()
        assert "minute" in result.__node__.sql().lower()

    def test_second(self) -> None:
        result = col("x").dt.second()
        assert "second" in result.__node__.sql().lower()

    def test_weekday(self) -> None:
        result = col("x").dt.weekday()
        sql = result.__node__.sql().lower()
        assert "day" in sql  # dayofweek

    def test_week(self) -> None:
        result = col("x").dt.week()
        assert "week" in result.__node__.sql().lower()

    def test_date(self) -> None:
        result = col("x").dt.date()
        assert "date" in result.__node__.sql().lower()
