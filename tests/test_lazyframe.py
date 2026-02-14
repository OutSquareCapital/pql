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
    qry = "SELECT 1 as a, 2 as b"
    assert_eq(pql.LazyFrame(duckdb.sql(qry)).collect(), duckdb.sql(qry).pl())


def test_lazyframe_from_pl_lazyframe(sample_df: pl.DataFrame) -> None:
    assert_eq(pql.LazyFrame(sample_df.lazy()).collect(), sample_df.lazy().collect())


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
    assert isinstance(pql.LazyFrame(sample_df).relation, pql.sql.Relation)


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
        pql.LazyFrame(sample_df).select("name", "age", "salary", "id").collect(),
        sample_df.lazy().select("name", "age", "salary", "id").collect(),
    )

    assert_eq(
        pql.LazyFrame(sample_df)
        .select(
            pql.col("name"),
            pql.col("salary").mul(1.1).alias("salary_increase"),
            vals=pql.col("id"),
            other_vals=42,
        )
        .collect(),
        sample_df.lazy()
        .select(
            pl.col("name"),
            pl.col("salary").mul(1.1).alias("salary_increase"),
            vals=pl.col("id"),
            other_vals=42,
        )
        .collect(),
    )


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
        .sort(pql.col("department"), "age", descending=[False, True])
        .collect(),
        sample_df.lazy()
        .sort(pl.col("department"), "age", descending=[False, True])
        .collect(),
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


def test_filter(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).filter(pql.col("salary").mul(12).gt(600000)).collect(),
        sample_df.lazy().filter(pl.col("salary").mul(12).gt(600000)).collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df)
        .filter(pql.col("salary").mul(12).gt(600000), pql.col("age").lt(50))
        .collect(),
        sample_df.lazy()
        .filter(pl.col("salary").mul(12).gt(600000), pl.col("age").lt(50))
        .collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df)
        .filter([pql.col("salary").mul(12).gt(600000), pql.col("age").lt(50)])
        .collect(),
        sample_df.lazy()
        .filter([pl.col("salary").mul(12).gt(600000), pl.col("age").lt(50)])
        .collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df)
        .filter(pql.col("age").gt(20), is_active=True, department="Sales")
        .collect(),
        sample_df.lazy()
        .filter(pl.col("age").gt(20), is_active=True, department="Sales")
        .collect(),
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


def test_top_k(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).top_k(3, by="age").collect()
    expected = sample_df.lazy().top_k(3, by="age").collect()
    assert_eq(result, expected)


def test_bottom_k(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).bottom_k(3, by="age").collect(),
        sample_df.lazy().bottom_k(3, by="age").collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df).bottom_k(3, by=["age", "salary"]).collect(),
        sample_df.lazy().bottom_k(3, by=["age", "salary"]).collect(),
    )


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
    assert_eq(
        (
            pql.LazyFrame(sample_df)
            .with_columns(pql.col("age").mul(2).alias("age_doubled"), x=42)
            .collect()
        ),
        (
            sample_df.lazy()
            .with_columns(pl.col("age").mul(2).alias("age_doubled"), x=42)
            .collect()
        ),
    )

    assert_eq(
        (
            pql.LazyFrame(sample_df)
            .with_columns(
                pql.col("age").mul(2).alias("age_doubled"),
                pql.col("salary").truediv(12).alias("monthly_salary"),
            )
            .collect()
        ),
        (
            sample_df.lazy()
            .with_columns(
                (pl.col("age") * 2).alias("age_doubled"),
                (pl.col("salary") / 12).alias("monthly_salary"),
            )
            .collect()
        ),
    )


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
