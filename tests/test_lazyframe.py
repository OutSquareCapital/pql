"""Tests for pql.LazyFrame class."""

from __future__ import annotations

import polars as pl
import pytest

import pql


class TestLazyFrameCreation:
    """Tests for creating LazyFrames."""

    def test_scan_table(self) -> None:
        sql = pql.LazyFrame().sql()
        assert "_" in sql.lower()
        assert "select" in sql.lower()

    def test_repr(self) -> None:
        lf = pql.LazyFrame()
        assert "LazyFrame" in repr(lf)

    def test_from_lazyframe(self) -> None:
        data = {"x": [1, 2, 3], "y": ["a", "b", "c"], "z": [True, False, True]}
        df = pql.LazyFrame(pl.DataFrame(data).lazy().select("x", "y")).collect()
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" not in df.columns


class TestSelect:
    """Tests for select operation."""

    def test_select_single_column(self) -> None:
        sql = pql.LazyFrame().select("x").sql().lower()
        assert "select" in sql
        assert "x" in sql

    def test_select_multiple_columns(self) -> None:
        sql = pql.LazyFrame().select("x", "y", "z").sql().lower()
        assert "x" in sql
        assert "y" in sql
        assert "z" in sql

    def test_select_with_expr(self) -> None:
        sql = pql.LazyFrame().select(pql.col("x").add(1)).sql().lower()
        assert "x + 1" in sql or "x+1" in sql

    def test_select_with_alias(self) -> None:
        sql = (
            pql.LazyFrame().select((pql.col("x").mul(2)).alias("doubled")).sql().lower()
        )
        assert "doubled" in sql


class TestWithColumns:
    """Tests for with_columns operation."""

    def test_with_columns_adds_column(self) -> None:
        sql = pql.LazyFrame().with_columns(pql.col("x").alias("x2")).sql().lower()
        assert "*" in sql
        assert "x2" in sql

    def test_with_columns_multiple(self) -> None:
        sql = (
            pql.LazyFrame()
            .with_columns(
                pql.col("x").add(1).alias("x_plus"),
                pql.col("y").mul(2).alias("y_times"),
            )
            .sql()
            .lower()
        )
        assert "x_plus" in sql
        assert "y_times" in sql


class TestFilter:
    """Tests for filter operation."""

    def test_filter_simple(self) -> None:
        sql = pql.LazyFrame().filter(pql.col("x").gt(5)).sql().lower()
        assert "where" in sql
        assert "x > 5" in sql

    def test_filter_multiple_predicates(self) -> None:
        sql = (
            pql.LazyFrame()
            .filter(pql.col("x").gt(5), pql.col("y").lt(10))
            .sql()
            .lower()
        )
        assert "where" in sql
        assert "and" in sql

    def test_filter_combined_with_and(self) -> None:
        sql = (
            pql.LazyFrame()
            .filter((pql.col("x").gt(5)).and_(pql.col("y").lt(10)))
            .sql()
            .lower()
        )
        assert "where" in sql
        assert "and" in sql

    def test_filter_combined_with_or(self) -> None:
        sql = (
            pql.LazyFrame()
            .filter((pql.col("x").gt(5)).or_(pql.col("y").lt(10)))
            .sql()
            .lower()
        )
        assert "where" in sql
        assert "or" in sql


class TestGroupBy:
    """Tests for group_by operation."""

    def test_group_by_single_column(self) -> None:
        sql = (
            pql.LazyFrame()
            .group_by("category")
            .agg(pql.col("value").sum().alias("total"))
            .sql()
            .lower()
        )
        assert "group by" in sql
        assert "category" in sql
        assert "sum" in sql

    def test_group_by_multiple_columns(self) -> None:
        sql = (
            pql.LazyFrame()
            .group_by("a", "b")
            .agg(pql.col("value").mean().alias("avg_val"))
            .sql()
            .lower()
        )
        assert "group by" in sql
        assert "a" in sql
        assert "b" in sql

    def test_group_by_multiple_aggs(self) -> None:
        sql = (
            pql.LazyFrame()
            .group_by("cat")
            .agg(
                pql.col("x").sum().alias("sum_x"),
                pql.col("y").mean().alias("avg_y"),
                pql.col("z").count().alias("cnt_z"),
            )
            .sql()
            .lower()
        )
        assert "sum" in sql
        assert "avg" in sql
        assert "count" in sql


class TestSort:
    """Tests for sort operation."""

    def test_sort_single_column(self) -> None:
        sql = pql.LazyFrame().sort("x").sql().lower()
        assert "order by" in sql
        assert "x" in sql

    def test_sort_descending(self) -> None:
        sql = pql.LazyFrame().sort("x", descending=True).sql().lower()
        assert "order by" in sql
        assert "desc" in sql

    def test_sort_multiple_columns(self) -> None:
        sql = pql.LazyFrame().sort("x", "y", descending=[True, False]).sql().lower()
        assert "order by" in sql

    def test_sort_nulls_first(self) -> None:
        # DuckDB defaults to NULLS LAST for ASC, so setting nulls_last=False
        # (i.e. nulls_first=True) should emit "NULLS FIRST" explicitly
        sql = pql.LazyFrame().sort("x", nulls_last=False).sql().lower()
        assert "order by" in sql
        assert "nulls first" in sql

    def test_sort_nulls_last_is_default(self) -> None:
        # DuckDB defaults to NULLS LAST for ASC, so nulls_last=True
        # won't emit anything explicit (it's the default behavior)
        sql = pql.LazyFrame().sort("x", nulls_last=True).sql().lower()
        assert "order by" in sql
        assert "x asc" in sql


class TestLimit:
    """Tests for limit operation."""

    def test_limit(self) -> None:
        sql = pql.LazyFrame().limit(10).sql().lower()
        assert "limit" in sql
        assert "10" in sql

    def test_head(self) -> None:
        sql = pql.LazyFrame().head(5).sql().lower()
        assert "limit" in sql
        assert "5" in sql

    def test_tail(self) -> None:
        # Note: tail in SQL context just uses limit
        sql = pql.LazyFrame().tail(5).sql().lower()
        assert "limit" in sql


class TestDistinct:
    """Tests for distinct operation."""

    def test_distinct(self) -> None:
        sql = pql.LazyFrame().distinct().sql().lower()
        assert "distinct" in sql

    def test_unique_no_subset(self) -> None:
        sql = pql.LazyFrame().unique().sql().lower()
        assert "distinct" in sql

    def test_unique_with_subset(self) -> None:
        assert "distinct" in pql.LazyFrame().unique(subset=["x", "y"]).sql().lower()


class TestDrop:
    """Tests for drop operation."""

    def test_drop_single_column(self) -> None:
        sql = pql.LazyFrame().drop("x").sql().lower()
        assert "except" in sql or "exclude" in sql or "x" in sql

    def test_drop_multiple_columns(self) -> None:
        sql = pql.LazyFrame().drop("x", "y").sql().lower()
        assert "*" in sql


class TestJoin:
    """Tests for join operation."""

    @pytest.fixture
    def left_df(self) -> pl.DataFrame:
        return pl.DataFrame({"id": [1, 2], "left_val": ["a", "b"]})

    @pytest.fixture
    def right_df(self) -> pl.DataFrame:
        return pl.DataFrame({"id": [2, 3], "right_val": ["x", "y"]})

    @pytest.fixture
    def right_df_alt(self) -> pl.DataFrame:
        return pl.DataFrame({"id2": [2, 3], "right_val": ["x", "y"]})

    def test_inner_join_on(self, left_df: pl.DataFrame, right_df: pl.DataFrame) -> None:
        result = pql.LazyFrame(left_df).join(pql.LazyFrame(right_df), on="id").collect()
        assert len(result) == 1
        assert result["id"][0] == 2

    def test_left_join(self, left_df: pl.DataFrame, right_df: pl.DataFrame) -> None:
        result = (
            pql.LazyFrame(left_df)
            .join(pql.LazyFrame(right_df), on="id", how="left")
            .collect()
        )
        assert len(result) == 2

    def test_right_join(self, left_df: pl.DataFrame, right_df: pl.DataFrame) -> None:
        result = (
            pql.LazyFrame(left_df)
            .join(pql.LazyFrame(right_df), on="id", how="right")
            .collect()
        )
        assert len(result) == 2

    def test_outer_join(self, left_df: pl.DataFrame, right_df: pl.DataFrame) -> None:
        result = (
            pql.LazyFrame(left_df)
            .join(pql.LazyFrame(right_df), on="id", how="outer")
            .collect()
        )
        assert len(result) == 3

    def test_join_left_on_right_on(
        self, left_df: pl.DataFrame, right_df_alt: pl.DataFrame
    ) -> None:
        result = (
            pql.LazyFrame(left_df)
            .join(pql.LazyFrame(right_df_alt), left_on="id", right_on="id2")
            .collect()
        )
        assert len(result) == 1


class TestChaining:
    """Tests for chaining multiple operations."""

    def test_filter_then_select(self) -> None:
        sql = (
            pql.LazyFrame()
            .filter(pql.col("age").gt(18))
            .select("name", "email")
            .sql()
            .lower()
        )
        assert "where" in sql
        assert "age > 18" in sql
        assert "name" in sql
        assert "email" in sql

    def test_full_pipeline(self) -> None:
        sql = (
            pql.LazyFrame()
            .filter(pql.col("status").eq("completed"))
            .group_by("customer_id")
            .agg(
                pql.col("amount").sum().alias("total_amount"),
                pql.col("order_id").count().alias("order_count"),
            )
            .sort("total_amount", descending=True)
            .limit(10)
            .sql()
            .lower()
        )
        assert "where" in sql
        assert "group by" in sql
        assert "sum" in sql
        assert "count" in sql
        assert "order by" in sql
        assert "desc" in sql
        assert "limit" in sql


class TestSqlOutput:
    """Tests for SQL output methods."""

    def test_sql_default_dialect(self) -> None:
        sql = pql.LazyFrame().sql()
        assert isinstance(sql, str)
        assert "SELECT" in sql

    def test_sql_pretty(self) -> None:
        lf = pql.LazyFrame().filter(pql.col("x").gt(5)).select("x", "y")
        # Pretty should have more newlines
        assert lf.sql(pretty=True).count("\n") >= lf.sql(pretty=False).count("\n")
