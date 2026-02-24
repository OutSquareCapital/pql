from __future__ import annotations

from functools import partial

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


def test_agg(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .group_by("department")
        .agg(pql.col("salary").mean().alias("mean_salary"))
        .sort("department")
        .collect(),
        sample_df.lazy()
        .group_by("department")
        .agg(pl.col("salary").mean().alias("mean_salary"))
        .sort("department")
        .collect(),
    )


def test_agg_by_prefix(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .group_by("department")
        .agg(pql.col("salary").mean().name.prefix("avg_"))
        .sort("department")
        .collect(),
        sample_df.lazy()
        .group_by("department")
        .agg(pl.col("salary").mean().name.prefix("avg_"))
        .sort("department")
        .collect(),
    )


@pytest.mark.parametrize(
    "method_name",
    ["all", "sum", "mean", "median", "min", "max", "first", "last", "n_unique"],
)
def test_lazygroupby_simple_computations(
    sample_df: pl.DataFrame, method_name: str
) -> None:
    selected = ("department", "age", "salary")
    result = (
        getattr(
            pql.LazyFrame(sample_df).select(*selected).group_by("department"),
            method_name,
        )()
        .sort("department")
        .collect()
    )
    expected = (
        getattr(
            sample_df.lazy().select(*selected).group_by("department"), method_name
        )()
        .sort("department")
        .collect()
    )
    assert_eq(result, expected)


def test_len(sample_df: pl.DataFrame) -> None:
    selected = ("department", "age", "salary")
    assert_eq(
        pql.LazyFrame(sample_df)
        .select(*selected)
        .group_by("department")
        .len()
        .sort("department")
        .collect(),
        sample_df.lazy()
        .select(*selected)
        .group_by("department")
        .len()
        .sort("department")
        .collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df)
        .select(*selected)
        .group_by("department")
        .len(name="n_rows")
        .sort("department")
        .collect(),
        sample_df.lazy()
        .select(*selected)
        .group_by("department")
        .len(name="n_rows")
        .sort("department")
        .collect(),
    )


@pytest.mark.parametrize(
    "interpolation", ["nearest", "linear", "lower", "higher", "midpoint"]
)
def test_quantile(interpolation: str) -> None:

    qdf = pl.DataFrame(
        {
            "department": ["A", "A", "A", "B", "B", "B"],
            "age": [10, 30, 50, 5, 25, 45],
            "salary": [100.0, 300.0, 500.0, 50.0, 250.0, 450.0],
        }
    )
    assert_eq(
        pql.LazyFrame(qdf)
        .group_by("department")
        .quantile(0.5, interpolation=interpolation)  # pyright: ignore[reportArgumentType]
        .sort("department")
        .collect(),
        qdf.lazy()
        .group_by("department")
        .quantile(0.5, interpolation=interpolation)  # pyright: ignore[reportArgumentType]
        .sort("department")
        .collect(),
    )
