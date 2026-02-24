from __future__ import annotations

from functools import partial

import polars as pl
import pyochain as pc
import pytest
from polars.testing import assert_frame_equal

import pql
import pql._typing

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


def test_properties(sample_df: pl.DataFrame) -> None:
    lf = pql.LazyFrame(sample_df)
    assert lf.width == sample_df.width
    assert set(lf.schema.keys()) == set(sample_df.columns)
    assert lf.schema == lf.collect_schema()
    assert isinstance(lf.lazy(), pl.LazyFrame)


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


def test_with_columns_name_prefix(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .with_columns(pql.col("name").name.prefix("new_"))
        .collect(),
        sample_df.lazy().with_columns(pl.col("name").name.prefix("new_")).collect(),
    )


def test_select_unique_name_prefix(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .select(pql.col("department").unique().name.prefix("u_"))
        .collect(),
        sample_df.lazy()
        .select(pl.col("department").unique().name.prefix("u_"))
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
    with pytest.raises(ValueError, match="length of `descending`"):
        _ = pql.LazyFrame(sample_df).sort("age", "salary", descending=[True]).collect()

    with pytest.raises(ValueError, match="length of `nulls_last`"):
        _ = (
            pql.LazyFrame(sample_df)
            .sort("age", "salary", nulls_last=[True, False, True])
            .collect()
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
        .cast({"age": pql.Float64()})
        .collect(),
        sample_df.lazy()
        .select(pl.col("age"), pl.col("id"))
        .cast({"age": pl.Float64})
        .collect(),
    )
    assert_frame_equal(
        pql.LazyFrame(sample_df)
        .select(pql.col("age"), pql.col("id"))
        .cast(pql.String())
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


def test_fill_null_with_value(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).select("value", "age").fill_null(0).collect(),
        sample_df.lazy().select("value", "age").fill_null(0).collect(),
    )


@pytest.mark.parametrize(
    "strategy",
    ["forward", "backward", "min", "max", "mean", "zero", "one"],
)
def test_fill_null_with_strategy(strategy: pql._typing.FillNullStrategy) -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0, None]})
    assert_eq(
        pql.LazyFrame(df).fill_null(strategy=strategy).collect(),
        df.lazy().fill_null(strategy=strategy).collect(),
    )


@pytest.mark.parametrize(
    "strategy",
    ["forward", "backward"],
)
def test_fill_null_with_strategy_limit(strategy: pql._typing.FillNullStrategy) -> None:
    df = pl.DataFrame({"a": [1, None, None, 4, None]})
    assert_eq(
        pql.LazyFrame(df).fill_null(strategy=strategy, limit=1).collect(),
        df.lazy().fill_null(strategy=strategy, limit=1).collect(),
    )


def test_fill_null_with_value_limit_error() -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
    with pytest.raises(ValueError, match="can only specify `limit`"):
        pql.LazyFrame(df).fill_null(0, limit=1).collect()


@pytest.mark.parametrize("strategy", ["min", "max", "mean", "zero", "one"])
def test_fill_null_with_non_directional_strategy_limit_error(
    strategy: pql._typing.FillNullStrategy,
) -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
    with pytest.raises(ValueError, match="can only specify `limit`"):
        pql.LazyFrame(df).fill_null(strategy=strategy, limit=1).collect()


def test_fill_null_with_negative_limit_error() -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
    with pytest.raises(
        pc.ResultUnwrapError, match="Can't process negative `limit` value for fill_null"
    ):
        pql.LazyFrame(df).fill_null(strategy="forward", limit=-1).collect()
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        df.lazy().fill_null(strategy="forward", limit=-1).collect()


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
    assert_eq(
        pql.LazyFrame(sample_df).top_k(3, by=["department", "age"]).collect(),
        sample_df.lazy().top_k(3, by=["department", "age"]).collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df).top_k(2, by="age", reverse=True).collect(),
        sample_df.lazy().top_k(2, by="age", reverse=True).collect(),
    )
    assert_eq(
        (
            pql.LazyFrame(sample_df)
            .top_k(3, by=["department", "age"], reverse=[False, True])
            .collect()
        ),
        (
            sample_df.lazy()
            .top_k(3, by=["department", "age"], reverse=[False, True])
            .collect()
        ),
    )


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


def test_lazyframe_drop_nulls(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).drop_nulls().collect(),
        sample_df.lazy().drop_nulls().collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df).drop_nulls("value").collect(),
        sample_df.lazy().drop_nulls("value").collect(),
    )


def test_lazyframe_explode() -> None:
    data = pl.DataFrame({"id": [1, 2, 3], "vals": [[10, 11], None, []]})
    assert_eq(
        pql.LazyFrame(data).explode("vals").collect(),
        data.lazy().explode("vals").collect(),
    )
    data = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "vals1": [[10, 11], [], None, [70]],
            "vals2": [[100, 110], [], None, [700]],
        }
    )
    assert_eq(
        pql.LazyFrame(data).explode("vals1", "vals2").collect(),
        data.lazy().explode("vals1", "vals2").collect(),
    )


def test_lazyframe_gather_every(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).gather_every(2, offset=1).collect(),
        sample_df.lazy().gather_every(2, offset=1).collect(),
    )


def test_lazyframe_slice(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).sort("id").slice(1, 3).collect(),
        sample_df.lazy().sort("id").slice(1, 3).collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df).sort("id").slice(2).collect(),
        sample_df.lazy().sort("id").slice(2).collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df).sort("id").slice(-2, 1).collect(),
        sample_df.lazy().sort("id").slice(-2, 1).collect(),
    )


def test_lazyframe_tail_and_last(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).sort("id").tail().collect(),
        sample_df.lazy().sort("id").tail().collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df).sort("id").tail(2).collect(),
        sample_df.lazy().sort("id").tail(2).collect(),
    )
    assert_eq(
        pql.LazyFrame(sample_df).sort("id").last().collect(),
        sample_df.lazy().sort("id").last().collect(),
    )


def test_lazyframe_describe(sample_df: pl.DataFrame) -> None:
    assert (
        pql.LazyFrame(sample_df).select("age", "salary").describe().collect().height > 0
    )


def test_lazyframe_join() -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    right = pl.DataFrame({"id": [2, 3, 4], "b": [200, 300, 400]})
    assert_eq(
        pql.LazyFrame(left).join(pql.LazyFrame(right), on="id", how="inner").collect(),
        left.lazy().join(right.lazy(), on="id", how="inner").collect(),
    )


def test_lazyframe_join_asof_backward() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "g2": ["x", "x", "y"], "b": [100, 200, 300]})
    assert_eq(
        pql.LazyFrame(left)
        .join_asof(
            pql.LazyFrame(right),
            left_on="t",
            right_on="u",
            by_left="g",
            by_right="g2",
            strategy="backward",
        )
        .collect(),
        left.lazy()
        .join_asof(
            right.lazy(),
            left_on="t",
            right_on="u",
            by_left="g",
            by_right="g2",
            strategy="backward",
        )
        .collect(),
    )


def test_lazyframe_join_left() -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    right = pl.DataFrame({"id": [2, 3, 4], "b": [200, 300, 400]})
    assert_eq(
        pql.LazyFrame(left).join(pql.LazyFrame(right), on="id", how="left").collect(),
        left.lazy().join(right.lazy(), on="id", how="left").collect(),
    )


def test_lazyframe_join_full() -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    right = pl.DataFrame({"id": [2, 3, 4], "b": [200, 300, 400]})
    assert_eq(
        pql.LazyFrame(left).join(pql.LazyFrame(right), on="id", how="full").collect(),
        left.lazy().join(right.lazy(), on="id", how="full").collect(),
    )


def test_lazyframe_join_cross() -> None:
    left = pl.DataFrame({"a": [1, 2]})
    right = pl.DataFrame({"b": [10, 20]})
    assert_eq(
        pql.LazyFrame(left).join(pql.LazyFrame(right), how="cross").collect(),
        left.lazy().join(right.lazy(), how="cross").collect(),
    )


def test_lazyframe_join_cross_with_keys_error() -> None:
    left = pl.DataFrame({"id": [1, 2], "a": [10, 20]})
    right = pl.DataFrame({"id": [2, 3], "b": [200, 300]})
    with pytest.raises(ValueError, match="Can not pass"):
        pql.LazyFrame(left).join(pql.LazyFrame(right), on="id", how="cross").collect()


def test_lazyframe_join_left_on_right_on() -> None:
    left = pl.DataFrame({"lid": [1, 2, 3], "a": [10, 20, 30]})
    right = pl.DataFrame({"rid": [2, 3, 4], "b": [200, 300, 400]})
    assert_eq(
        pql.LazyFrame(left)
        .join(pql.LazyFrame(right), left_on="lid", right_on="rid", how="inner")
        .collect(),
        left.lazy()
        .join(right.lazy(), left_on="lid", right_on="rid", how="inner")
        .collect(),
    )


def test_lazyframe_join_semi() -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    right = pl.DataFrame({"id": [2, 3, 4], "b": [200, 300, 400]})
    result = (
        pql.LazyFrame(left).join(pql.LazyFrame(right), on="id", how="semi").collect()
    )
    expected = left.lazy().join(right.lazy(), on="id", how="semi").collect()
    assert_eq(result, expected)


def test_lazyframe_join_anti() -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    right = pl.DataFrame({"id": [2, 3, 4], "b": [200, 300, 400]})
    result = (
        pql.LazyFrame(left).join(pql.LazyFrame(right), on="id", how="anti").collect()
    )
    expected = left.lazy().join(right.lazy(), on="id", how="anti").collect()
    assert_eq(result, expected)


def test_lazyframe_join_on_multiple_keys() -> None:
    left = pl.DataFrame({"id1": [1, 2, 3], "id2": ["a", "b", "c"], "a": [10, 20, 30]})
    right = pl.DataFrame(
        {"id1": [2, 3, 4], "id2": ["b", "c", "d"], "b": [200, 300, 400]}
    )
    assert_eq(
        pql.LazyFrame(left)
        .join(pql.LazyFrame(right), on=["id1", "id2"], how="inner")
        .collect(),
        left.lazy().join(right.lazy(), on=["id1", "id2"], how="inner").collect(),
    )


def test_lazyframe_join_column_overlap() -> None:
    left = pl.DataFrame({"id": [1, 2], "a": [10, 20]})
    right = pl.DataFrame({"id": [1, 2], "a": [100, 200]})
    result = pql.LazyFrame(left).join(pql.LazyFrame(right), on="id").collect()
    expected = left.lazy().join(right.lazy(), on="id").collect()
    assert_eq(result, expected)


def test_lazyframe_join_asof_forward() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "g2": ["x", "x", "y"], "b": [100, 200, 300]})
    assert_eq(
        pql.LazyFrame(left)
        .join_asof(
            pql.LazyFrame(right),
            left_on="t",
            right_on="u",
            by_left="g",
            by_right="g2",
            strategy="forward",
        )
        .collect(),
        left.lazy()
        .join_asof(
            right.lazy(),
            left_on="t",
            right_on="u",
            by_left="g",
            by_right="g2",
            strategy="forward",
        )
        .collect(),
    )


def test_lazyframe_join_asof_nearest() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "g2": ["x", "x", "y"], "b": [100, 200, 300]})
    assert_eq(
        pql.LazyFrame(left)
        .join_asof(
            pql.LazyFrame(right),
            left_on="t",
            right_on="u",
            by_left="g",
            by_right="g2",
            strategy="nearest",
        )
        .collect(),
        left.lazy()
        .join_asof(
            right.lazy(),
            left_on="t",
            right_on="u",
            by_left="g",
            by_right="g2",
            strategy="nearest",
        )
        .collect(),
    )


def test_lazyframe_join_asof_error_on_and_left_on() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "b": [100, 200, 300]})
    with pytest.raises(ValueError, match="If `on` is specified"):
        pql.LazyFrame(left).join_asof(
            pql.LazyFrame(right), on="t", left_on="t", right_on="u"
        ).collect()


def test_lazyframe_join_asof_error_no_keys() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "b": [100, 200, 300]})
    with pytest.raises(ValueError, match="Either"):
        pql.LazyFrame(left).join_asof(pql.LazyFrame(right)).collect()


def test_lazyframe_join_asof_error_left_on_without_right_on() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "b": [100, 200, 300]})
    with pytest.raises(ValueError, match="Either"):
        pql.LazyFrame(left).join_asof(pql.LazyFrame(right), left_on="t").collect()


def test_lazyframe_join_asof_error_by_and_by_left() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "g2": ["x", "x", "y"], "b": [100, 200, 300]})
    with pytest.raises(ValueError, match="If `by` is specified"):
        pql.LazyFrame(left).join_asof(
            pql.LazyFrame(right),
            left_on="t",
            right_on="u",
            by="g",
            by_left="g",
        ).collect()


def test_lazyframe_join_asof_error_by_left_without_by_right() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "g2": ["x", "x", "y"], "b": [100, 200, 300]})
    with pytest.raises(ValueError, match="Can not specify only"):
        pql.LazyFrame(left).join_asof(
            pql.LazyFrame(right),
            left_on="t",
            right_on="u",
            by_left="g",
        ).collect()


def test_lazyframe_join_asof_error_unequal_by_lengths() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "g2": ["x", "x", "y"], "b": [100, 200, 300]})
    with pytest.raises(ValueError, match="must have the same length"):
        pql.LazyFrame(left).join_asof(
            pql.LazyFrame(right),
            left_on="t",
            right_on="u",
            by_left="g",
            by_right=["g2", "b"],
        ).collect()


def test_lazyframe_unique_any(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).unique(subset=["department"]).collect(),
        sample_df.lazy().unique(subset=["department"], keep="any").collect(),
    )


def test_lazyframe_unique_first(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .unique(subset=["department"], keep="first", order_by="id")
        .collect(),
        sample_df.lazy().unique(subset=["department"], keep="first").collect(),
    )


def test_lazyframe_unique_last(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df)
        .unique(subset=["department"], keep="last", order_by="id")
        .collect(),
        sample_df.lazy().unique(subset=["department"], keep="last").collect(),
    )


def test_lazyframe_unique_none(sample_df: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df).unique(subset=["department"], keep="none").collect()
    )
    assert (
        result.height
        == sample_df.lazy().unique(subset=["department"], keep="none").collect().height
    )


def test_lazyframe_unique_first_without_order_by_error() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
    with pytest.raises(ValueError, match="`order_by` must be specified"):
        pql.LazyFrame(df).unique(keep="first").collect()


def test_lazyframe_unique_last_without_order_by_error() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
    with pytest.raises(ValueError, match="`order_by` must be specified"):
        pql.LazyFrame(df).unique(keep="last").collect()


def test_lazyframe_unique_with_multiple_order_by() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [10, 20, 30]})
    result = pql.LazyFrame(df).unique(keep="first", order_by=["a", "b"]).collect()
    assert result.height > 0


def test_lazyframe_unpivot() -> None:
    data = pl.DataFrame({"id": ["a", "b"], "x": [1, 3], "y": [2, 4]})
    assert_eq(
        pql.LazyFrame(data).unpivot(on=["x", "y"], index="id").collect(),
        data.lazy().unpivot(on=["x", "y"], index="id").collect(),
    )


def test_select_with_named_expr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = (
        pql.LazyFrame(df).select(pql.col("a"), doubled=pql.col("b").mul(2)).collect()
    )
    expected = df.lazy().select(pl.col("a"), doubled=pl.col("b").mul(2)).collect()
    assert_eq(result, expected)


def test_join_left_on_right_on_length_mismatch() -> None:
    left = pl.DataFrame({"id1": [1, 2], "id2": ["a", "b"], "a": [10, 20]})
    right = pl.DataFrame({"id1": [1, 2], "b": [100, 200]})
    with pytest.raises(ValueError, match="same length"):
        pql.LazyFrame(left).join(
            pql.LazyFrame(right), left_on=["id1", "id2"], right_on="id1", how="inner"
        ).collect()


def test_join_left_with_multiple_keys() -> None:
    left = pl.DataFrame({"id1": [1, 2, 3], "id2": ["a", "b", "c"], "a": [10, 20, 30]})
    right = pl.DataFrame(
        {"id1": [2, 3, 4], "id2": ["b", "c", "d"], "b": [200, 300, 400]}
    )
    result = (
        pql.LazyFrame(left)
        .join(
            pql.LazyFrame(right),
            left_on=["id1", "id2"],
            right_on=["id1", "id2"],
            how="left",
        )
        .collect()
    )
    expected = (
        left.lazy()
        .join(
            right.lazy(),
            left_on=["id1", "id2"],
            right_on=["id1", "id2"],
            how="left",
        )
        .collect()
    )
    assert_eq(result, expected)


def test_lazyframe_quantile(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df).select("age", "salary").quantile(0.5).collect(),
        sample_df.lazy().select("age", "salary").quantile(0.5).collect(),
    )


def test_lazyframe_join_without_keys_error() -> None:
    left = pl.DataFrame({"id": [1, 2], "a": [10, 20]})
    right = pl.DataFrame({"id": [1, 2], "b": [100, 200]})
    with pytest.raises(ValueError, match="Either"):
        pql.LazyFrame(left).join(pql.LazyFrame(right), how="inner").collect()


def test_lazyframe_join_on_and_left_right_on_error() -> None:
    left = pl.DataFrame({"id": [1, 2], "a": [10, 20]})
    right = pl.DataFrame({"id": [1, 2], "b": [100, 200]})
    with pytest.raises(ValueError, match="If `on` is specified"):
        (
            pql.LazyFrame(left)
            .join(
                pql.LazyFrame(right),
                on="id",
                left_on="id",
                right_on="id",
                how="inner",
            )
            .collect()
        )


def test_lazyframe_join_asof_on_without_by() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "a": [1, 2, 3]})
    right = pl.DataFrame({"t": [0, 3, 8], "b": [100, 200, 300]})
    assert_eq(
        pql.LazyFrame(left)
        .join_asof(pql.LazyFrame(right), on="t", strategy="backward")
        .collect(),
        left.lazy().join_asof(right.lazy(), on="t", strategy="backward").collect(),
    )


def test_lazyframe_join_asof_with_by() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
    right = pl.DataFrame({"t": [0, 3, 8], "g": ["x", "x", "y"], "b": [100, 200, 300]})
    assert_eq(
        pql.LazyFrame(left)
        .join_asof(
            pql.LazyFrame(right),
            on="t",
            by="g",
            strategy="backward",
        )
        .collect(),
        left.lazy()
        .join_asof(
            right.lazy(),
            on="t",
            by="g",
            strategy="backward",
        )
        .collect(),
    )


def test_lazyframe_join_asof_overlap_column_suffix() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "a": [1, 2, 3]})
    right = pl.DataFrame({"t": [0, 3, 8], "a": [100, 200, 300]})
    assert_eq(
        pql.LazyFrame(left)
        .join_asof(pql.LazyFrame(right), on="t", strategy="backward")
        .collect(),
        left.lazy().join_asof(right.lazy(), on="t", strategy="backward").collect(),
    )
