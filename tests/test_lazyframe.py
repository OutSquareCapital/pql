"""Tests for pql.LazyFrame class."""

from __future__ import annotations

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


class TestSelect:
    """Tests for select operation."""

    def test_select_single_column(self) -> None:
        lf = pql.LazyFrame().select("x")
        sql = lf.sql().lower()
        assert "select" in sql
        assert "x" in sql

    def test_select_multiple_columns(self) -> None:
        lf = pql.LazyFrame().select("x", "y", "z")
        sql = lf.sql().lower()
        assert "x" in sql
        assert "y" in sql
        assert "z" in sql

    def test_select_with_expr(self) -> None:
        lf = pql.LazyFrame().select(pql.col("x").add(1))
        sql = lf.sql().lower()
        assert "x + 1" in sql or "x+1" in sql

    def test_select_with_alias(self) -> None:
        lf = pql.LazyFrame().select((pql.col("x").mul(2)).alias("doubled"))
        sql = lf.sql().lower()
        assert "doubled" in sql


class TestWithColumns:
    """Tests for with_columns operation."""

    def test_with_columns_adds_column(self) -> None:
        lf = pql.LazyFrame().with_columns(pql.col("x").alias("x2"))
        sql = lf.sql().lower()
        assert "*" in sql
        assert "x2" in sql

    def test_with_columns_multiple(self) -> None:
        lf = pql.LazyFrame().with_columns(
            pql.col("x").add(1).alias("x_plus"),
            pql.col("y").mul(2).alias("y_times"),
        )
        sql = lf.sql().lower()
        assert "x_plus" in sql
        assert "y_times" in sql


class TestFilter:
    """Tests for filter operation."""

    def test_filter_simple(self) -> None:
        lf = pql.LazyFrame().filter(pql.col("x").gt(5))
        sql = lf.sql().lower()
        assert "where" in sql
        assert "x > 5" in sql

    def test_filter_multiple_predicates(self) -> None:
        lf = pql.LazyFrame().filter(pql.col("x").gt(5), pql.col("y").lt(10))
        sql = lf.sql().lower()
        assert "where" in sql
        assert "and" in sql

    def test_filter_combined_with_and(self) -> None:
        lf = pql.LazyFrame().filter((pql.col("x").gt(5)).and_(pql.col("y").lt(10)))
        sql = lf.sql().lower()
        assert "where" in sql
        assert "and" in sql

    def test_filter_combined_with_or(self) -> None:
        lf = pql.LazyFrame().filter((pql.col("x").gt(5)).or_(pql.col("y").lt(10)))
        sql = lf.sql().lower()
        assert "where" in sql
        assert "or" in sql


class TestGroupBy:
    """Tests for group_by operation."""

    def test_group_by_single_column(self) -> None:
        lf = (
            pql.LazyFrame()
            .group_by("category")
            .agg(pql.col("value").sum().alias("total"))
        )
        sql = lf.sql().lower()
        assert "group by" in sql
        assert "category" in sql
        assert "sum" in sql

    def test_group_by_multiple_columns(self) -> None:
        lf = (
            pql.LazyFrame()
            .group_by("a", "b")
            .agg(pql.col("value").mean().alias("avg_val"))
        )
        sql = lf.sql().lower()
        assert "group by" in sql
        assert "a" in sql
        assert "b" in sql

    def test_group_by_multiple_aggs(self) -> None:
        lf = (
            pql.LazyFrame()
            .group_by("cat")
            .agg(
                pql.col("x").sum().alias("sum_x"),
                pql.col("y").mean().alias("avg_y"),
                pql.col("z").count().alias("cnt_z"),
            )
        )
        sql = lf.sql().lower()
        assert "sum" in sql
        assert "avg" in sql
        assert "count" in sql


class TestSort:
    """Tests for sort operation."""

    def test_sort_single_column(self) -> None:
        lf = pql.LazyFrame().sort("x")
        sql = lf.sql().lower()
        assert "order by" in sql
        assert "x" in sql

    def test_sort_descending(self) -> None:
        lf = pql.LazyFrame().sort("x", descending=True)
        sql = lf.sql().lower()
        assert "order by" in sql
        assert "desc" in sql

    def test_sort_multiple_columns(self) -> None:
        lf = pql.LazyFrame().sort("x", "y", descending=[True, False])
        sql = lf.sql().lower()
        assert "order by" in sql

    def test_sort_nulls_first(self) -> None:
        # DuckDB defaults to NULLS LAST for ASC, so setting nulls_last=False
        # (i.e. nulls_first=True) should emit "NULLS FIRST" explicitly
        lf = pql.LazyFrame().sort("x", nulls_last=False)
        sql = lf.sql().lower()
        assert "order by" in sql
        assert "nulls first" in sql

    def test_sort_nulls_last_is_default(self) -> None:
        # DuckDB defaults to NULLS LAST for ASC, so nulls_last=True
        # won't emit anything explicit (it's the default behavior)
        lf = pql.LazyFrame().sort("x", nulls_last=True)
        sql = lf.sql().lower()
        assert "order by" in sql
        assert "x asc" in sql


class TestLimit:
    """Tests for limit operation."""

    def test_limit(self) -> None:
        lf = pql.LazyFrame().limit(10)
        sql = lf.sql().lower()
        assert "limit" in sql
        assert "10" in sql

    def test_head(self) -> None:
        lf = pql.LazyFrame().head(5)
        sql = lf.sql().lower()
        assert "limit" in sql
        assert "5" in sql

    def test_tail(self) -> None:
        # Note: tail in SQL context just uses limit
        lf = pql.LazyFrame().tail(5)
        sql = lf.sql().lower()
        assert "limit" in sql


class TestDistinct:
    """Tests for distinct operation."""

    def test_distinct(self) -> None:
        lf = pql.LazyFrame().distinct()
        sql = lf.sql().lower()
        assert "distinct" in sql

    def test_unique_no_subset(self) -> None:
        lf = pql.LazyFrame().unique()
        sql = lf.sql().lower()
        assert "distinct" in sql

    def test_unique_with_subset(self) -> None:
        assert "distinct" in pql.LazyFrame().unique(subset=["x", "y"]).sql().lower()


class TestDrop:
    """Tests for drop operation."""

    def test_drop_single_column(self) -> None:
        lf = pql.LazyFrame().drop("x")
        sql = lf.sql().lower()
        assert "except" in sql or "exclude" in sql or "x" in sql

    def test_drop_multiple_columns(self) -> None:
        lf = pql.LazyFrame().drop("x", "y")
        sql = lf.sql().lower()
        assert "*" in sql


class TestJoin:
    """Tests for join operation."""

    def test_inner_join_on(self) -> None:
        result = pql.LazyFrame().join(pql.LazyFrame(), on="id")
        sql = result.sql().lower()
        assert "join" in sql

    def test_left_join(self) -> None:
        lf1 = pql.LazyFrame()
        lf2 = pql.LazyFrame()
        result = lf1.join(lf2, on="id", how="left")
        sql = result.sql().lower()
        assert "left" in sql
        assert "join" in sql

    def test_right_join(self) -> None:
        lf1 = pql.LazyFrame()
        lf2 = pql.LazyFrame()
        result = lf1.join(lf2, on="id", how="right")
        sql = result.sql().lower()
        assert "right" in sql
        assert "join" in sql

    def test_outer_join(self) -> None:
        lf1 = pql.LazyFrame()
        lf2 = pql.LazyFrame()
        result = lf1.join(lf2, on="id", how="outer")
        sql = result.sql().lower()
        assert "full" in sql or "outer" in sql

    def test_join_left_on_right_on(self) -> None:
        lf1 = pql.LazyFrame()
        lf2 = pql.LazyFrame()
        result = lf1.join(lf2, left_on="id1", right_on="id2")
        sql = result.sql().lower()
        assert "join" in sql


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
        lf = (
            pql.LazyFrame()
            .filter(pql.col("status").eq("completed"))
            .group_by("customer_id")
            .agg(
                pql.col("amount").sum().alias("total_amount"),
                pql.col("order_id").count().alias("order_count"),
            )
            .sort("total_amount", descending=True)
            .limit(10)
        )
        sql = lf.sql().lower()
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
        sql_pretty = lf.sql(pretty=True)
        sql_compact = lf.sql(pretty=False)
        # Pretty should have more newlines
        assert sql_pretty.count("\n") >= sql_compact.count("\n")

    def test_explain(self) -> None:
        assert "EXPLAIN" in pql.LazyFrame().explain()
