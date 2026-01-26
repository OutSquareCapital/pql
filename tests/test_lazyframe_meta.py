from functools import partial

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pql

assert_eq = partial(assert_frame_equal, check_dtypes=False, check_row_order=False)


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


def test_lazyframe_from_duckdb_relation() -> None:
    import duckdb

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
    assert isinstance(lf.dtypes, list)
    assert lf.width == len(sample_df.columns)
    assert isinstance(lf.schema, dict)
    assert set(lf.schema.keys()) == set(sample_df.columns)
    assert lf.schema == lf.collect_schema()


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
