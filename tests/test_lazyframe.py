"""Tests for expression operations to achieve 100% coverage - comparing with Polars."""

from __future__ import annotations

from functools import partial

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pql

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
            "value": [10.0, None, 30.0, None, 50.0],
            "category": ["A", "B", None, "A", "B"],
        }
    )


def test_lazyframe_from_duckdb_relation() -> None:
    result = pql.LazyFrame(duckdb.sql("SELECT 1 as a, 2 as b")).collect()
    expected = duckdb.sql("SELECT 1 as a, 2 as b").pl()
    assert_eq(result, expected)


def test_lazyframe_from_pl_lazyframe(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df.lazy().filter(pl.col("age") > 20)).collect()
    expected = sample_df.lazy().filter(pl.col("age") > 20).collect()
    assert_eq(result, expected)


def test_lazyframe_from_empty_value() -> None:
    result = pql.LazyFrame().collect()
    assert result.height == 0
    assert result.width == 1


def test_lazyframe_from_dict() -> None:
    result = pql.LazyFrame({"a": [1, 2, 3]}).collect()
    expected = pl.DataFrame({"a": [1, 2, 3]})
    assert_eq(result, expected)


def test_properties(sample_df: pl.DataFrame) -> None:
    lf = pql.LazyFrame(sample_df)
    assert len(lf.dtypes) == len(sample_df.dtypes)
    assert lf.width == sample_df.width
    assert set(lf.schema.keys()) == set(sample_df.columns)
    assert lf.schema == lf.collect_schema()
    assert isinstance(lf.lazy(), pl.LazyFrame)


def test_relation_property(sample_df: pl.DataFrame) -> None:
    assert isinstance(pql.LazyFrame(sample_df).relation, duckdb.DuckDBPyRelation)


def test_repr(sample_df: pl.DataFrame) -> None:
    assert "LazyFrame" in repr(pql.LazyFrame(sample_df))


def test_clone(sample_df: pl.DataFrame) -> None:
    lf = pql.LazyFrame(sample_df)
    cloned = lf.clone()
    assert_eq(lf.collect(), cloned.collect())
    cloned_modified = cloned.filter(pql.col("age").gt(25))
    assert lf.collect().height != cloned_modified.collect().height


def test_explain(sample_df: pl.DataFrame) -> None:
    sql = (
        pql.LazyFrame(sample_df)
        .filter(pql.col("age").gt(25))
        .select("name", "age")
        .explain()
    )
    assert isinstance(sql, str)
    assert "SELECT" in sql
    assert "WHERE" in sql


def test_select_single_column(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).select("name").collect(),
        sample_df.lazy().select("name").collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df).select(pql.col("name")).collect(),
        sample_df.lazy().select(pl.col("name")).collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df).select(duckdb.ColumnExpression("name")).collect(),
        sample_df.lazy().select(pl.col("name")).collect(),
    )


def test_select_multiple_columns(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .select("name", "age", "salary", duckdb.ColumnExpression("id"))
        .collect(),
        sample_df.lazy().select("name", "age", "salary", "id").collect(),
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


def test_fill_null(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .select(pql.col("id"), pql.col("value"))
        .fill_null(0)
        .collect(),
        (sample_df.lazy().select(pl.col("id"), pl.col("value")).fill_null(0).collect()),
    )

    assert_eq(
        pql.LazyFrame(sample_df).fill_null(strategy="forward").collect(),
        sample_df.lazy().fill_null(strategy="forward").collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df).fill_null(strategy="backward").collect(),
        sample_df.lazy().fill_null(strategy="backward").collect(),
    )
    with pytest.raises(
        ValueError, match=r"Either `value` or `strategy` must be provided\."
    ):
        pql.LazyFrame(sample_df).fill_null().collect()


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
    assert_eq(
        pql.LazyFrame(sample_df).sort("age", nulls_last=True).collect(),
        sample_df.lazy().sort("age", nulls_last=True).collect(),
    )

    assert_eq(
        pql.LazyFrame(sample_df)
        .sort("age", "department", nulls_last=[True, False])
        .collect(),
        sample_df.lazy().sort("age", "department", nulls_last=[True, False]).collect(),
    )


def test_limit(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).sort("id").limit(3).collect(),
        sample_df.lazy().sort("id").limit(3).collect(),
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


def test_null_count(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("value")).null_count().collect()
    expected = sample_df.lazy().select(pl.col("value")).null_count().collect()
    assert_eq(result, expected, check_dtypes=False)


def test_drop_nulls(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).drop_nulls().collect()
    expected = sample_df.lazy().drop_nulls().collect()
    assert_eq(result, expected)


def test_drop_nulls_subset(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).drop_nulls(subset="value").collect()
    expected = sample_df.lazy().drop_nulls(subset="value").collect()
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


def test_top_k(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).top_k(3, by="age").collect()
    expected = sample_df.lazy().top_k(3, by="age").collect()
    assert_eq(result, expected)


def test_bottom_k(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).bottom_k(3, by="age").collect()
    expected = sample_df.lazy().bottom_k(3, by="age").collect()
    assert_eq(result, expected)


def test_cast(sample_df: pl.DataFrame) -> None:
    assert_frame_equal(
        pql.LazyFrame(sample_df)
        .select(pql.col("age"), pql.col("id"))
        .cast({"age": pql.Float64})
        .collect(),
        sample_df.lazy()
        .select(pl.col("age"), pl.col("id"))
        .cast({"age": pl.Float64})
        .collect(),
    )
    assert_frame_equal(
        pql.LazyFrame(sample_df)
        .select(pql.col("age"), pql.col("id"))
        .cast(pql.String)
        .collect(),
        sample_df.lazy().select(pl.col("age"), pl.col("id")).cast(pl.String).collect(),
    )


def test_pipe(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).pipe(lambda lf: lf).collect()
    expected = sample_df.lazy().pipe(lambda df: df).collect()
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
    df = pl.DataFrame(
        {
            "a": [1.0, float("nan"), 3.0],
            "b": [4.0, 5.0, float("nan")],
            "c": [7.0, 8.0, 9.0],
        }
    )
    assert_eq(pql.LazyFrame(df).drop_nans().collect(), df.lazy().drop_nans().collect())
    assert_eq(
        pql.LazyFrame(df).drop_nans(subset="a").collect(),
        df.lazy().drop_nans(subset="a").collect(),
    )
    assert_eq(
        pql.LazyFrame(df).drop_nans(subset=["a", "b"]).collect(),
        df.lazy().drop_nans(subset=["a", "b"]).collect(),
    )


def test_shift() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert_eq(pql.LazyFrame(df).shift(2).collect(), df.lazy().shift(2).collect())
    assert_eq(pql.LazyFrame(df).shift(-2).collect(), df.lazy().shift(-2).collect())

    assert_eq(
        pql.LazyFrame(df).shift(2, fill_value=0).collect(),
        df.lazy().shift(2, fill_value=0).collect(),
    )
    assert_eq(
        pql.LazyFrame(df).shift(1, fill_value=999).collect(),
        df.lazy().shift(1, fill_value=999).collect(),
    )


def test_is_unique() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    assert_eq(
        pql.LazyFrame(df)
        .with_columns(pql.col("a").is_unique().alias("is_unique"))
        .collect(),
        df.lazy().with_columns(pl.col("a").is_unique().alias("is_unique")).collect(),
    )
    assert_eq(
        pql.LazyFrame(df).unique(subset="a").collect(),
        df.lazy().unique(subset="a").collect(),
    )
    assert_eq(
        pql.LazyFrame(df).unique(subset=["a", "b"]).collect(),
        df.lazy().unique(subset=["a", "b"]).collect(),
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


def test_top_k_with_multiple_cols(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).top_k(3, by=["department", "age"]).collect()
    expected = sample_df.lazy().top_k(3, by=["department", "age"]).collect()
    assert_eq(result, expected)


def test_top_k_with_reverse(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).top_k(2, by="age", reverse=True).collect()
    expected = sample_df.lazy().top_k(2, by="age", reverse=True).collect()
    assert_eq(result, expected)


def test_bottom_k_with_multiple_cols(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).bottom_k(3, by=["department", "age"]).collect(),
        sample_df.lazy().bottom_k(3, by=["department", "age"]).collect(),
    )


def test_bottom_k_with_reverse(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).bottom_k(2, by="age", reverse=True).collect()
    expected = sample_df.lazy().bottom_k(2, by="age", reverse=True).collect()
    assert_eq(result, expected)


def test_hash_seed0() -> None:
    df = pl.DataFrame({"text": ["apple", "banana", "apple"]})
    result = pql.LazyFrame(df).select(pql.col("text").hash(seed=0).alias("h")).collect()
    # Check that same input produces same hash
    hashes = result["h"].to_list()
    assert hashes[0] == hashes[2], "Same input should produce same hash"


def test_hash_seed42() -> None:
    df = pl.DataFrame({"text": ["apple", "banana", "apple"]})
    result = (
        pql.LazyFrame(df).select(pql.col("text").hash(seed=42).alias("h")).collect()
    )
    # Check that same input produces same hash with different seed
    hashes = result["h"].to_list()
    assert hashes[0] == hashes[2], "Same input should produce same hash"
    # Different seed should produce different hash
    result_seed0 = (
        pql.LazyFrame(df).select(pql.col("text").hash(seed=0).alias("h")).collect()
    )
    assert hashes[0] != result_seed0["h"][0], (
        "Different seeds should produce different hashes"
    )
