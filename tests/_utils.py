from collections.abc import Iterable

import duckdb
import narwhals as nw
import polars as pl
from polars.testing import assert_frame_equal

import pql


def sample_df() -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    nan = float("nan")
    data = {
        "a": [True, False, True, None, True, False],
        "b": [True, True, False, None, True, False],
        "x": [10, 2, 3, 5, 10, 20],
        "n": [None, 3, 1, None, 2, 3],
        "s": ["1", "2", "3", None, "1", "2"],
        "age": [25, 30, 35, None, 25, 30],
        "salary": [50000.0, 60000.0, 70000.0, None, 50000.0, 60000.0],
        "nested": [[1, 2], [3, 4], [5], None, [1, 2], [3, 4]],
        "nan_vals": [1.0, nan, 3.0, nan, 5.0, nan],
        "structs": [
            {"a": 1, "b": 2, "c": 3, "d": 4},
            {"a": 5, "b": 6, "c": 7, "d": 8},
            {"a": 5, "b": 6, "c": 7, "d": 8},
            {"a": 5, "b": 6, "c": 7, "d": 8},
            {"a": 5, "b": 6, "c": 7, "d": 8},
            {"a": 5, "b": 6, "c": 7, "d": 8},
        ],
    }
    return nw.from_native(pl.DataFrame(data).pipe(duckdb.from_arrow))


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
