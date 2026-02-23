from collections.abc import Iterable

import duckdb
import narwhals as nw
import polars as pl
from polars.testing import assert_frame_equal

import pql


def sample_df() -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    return nw.from_native(
        duckdb.from_arrow(
            pl.DataFrame(
                {
                    "a": [True, False, True, None, True, False],
                    "b": [True, True, False, None, True, False],
                    "x": [10, 2, 3, 5, 10, 20],
                    "n": [2, 3, 1, None, 2, 3],
                    "s": ["1", "2", "3", None, "1", "2"],
                    "age": [25, 30, 35, None, 25, 30],
                    "salary": [50000.0, 60000.0, 70000.0, None, 50000.0, 60000.0],
                    "nested": [[1, 2], [3, 4], [5], None, [1, 2], [3, 4]],
                }
            )
        )
    )


def assert_eq(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: nw.Expr | Iterable[nw.Expr]
) -> None:
    assert_frame_equal(
        pql.LazyFrame(sample_df().to_native()).select(pql_exprs).collect(),
        sample_df().lazy().select(polars_exprs).to_native().pl(),
        check_dtypes=False,
        check_row_order=False,
    )


def assert_eq_pl(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: pl.Expr | Iterable[pl.Expr]
) -> None:
    assert_frame_equal(
        pql.LazyFrame(sample_df().to_native()).select(pql_exprs).collect(),
        sample_df().to_native().pl(lazy=True).select(polars_exprs).collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def test_name_keep_after_alias() -> None:
    assert_eq(
        pql.col("x").alias("renamed").name.keep(),
        nw.col("x").alias("renamed").name.keep(),
    )


def test_name_map() -> None:
    assert_eq(
        pql.col("x").name.map(lambda name: f"mapped_{name}"),
        nw.col("x").name.map(lambda name: f"mapped_{name}"),
    )


def test_name_prefix_suffix() -> None:
    assert_eq(
        pql.col("x").name.prefix("pre_").name.suffix("_suf"),
        nw.col("x").name.prefix("pre_").name.suffix("_suf"),
    )


def test_name_case_transform() -> None:
    assert_eq(
        pql.col("x").name.to_uppercase().name.to_lowercase(),
        nw.col("x").name.to_uppercase().name.to_lowercase(),
    )


def test_name_to_uppercase_all() -> None:
    assert_eq(
        pql.all().name.to_uppercase(),
        nw.all().name.to_uppercase(),
    )


def test_name_replace() -> None:
    assert_eq_pl(
        pql.col("x").name.replace("x", "y"),
        pl.col("x").name.replace("x", "y"),
    )
    assert_eq_pl(
        pql.col("s").name.replace("s", "t"),
        pl.col("s").name.replace("s", "t"),
    )
    assert_eq_pl(
        pql.col("salary").name.replace("salary", "income"),
        pl.col("salary").name.replace("salary", "income"),
    )

    # Test avec literal=True (remplacement exact, pas regex)
    assert_eq_pl(
        pql.col("salary").name.replace("a", "b", literal=True),
        pl.col("salary").name.replace("a", "b", literal=True),
    )
    assert_eq_pl(
        pql.col("salary").name.replace("l", "L", literal=True),
        pl.col("salary").name.replace("l", "L", literal=True),
    )
    assert_eq_pl(
        pql.col("salary").name.replace("sal", "SAL", literal=True),
        pl.col("salary").name.replace("sal", "SAL", literal=True),
    )
