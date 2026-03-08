from collections.abc import Iterable
from datetime import date, datetime, time

import duckdb
import narwhals as nw
import polars as pl
from polars.testing import assert_frame_equal

import pql

nan = float("nan")

_DATA = {
    "a": [True, False, True, None, True, False],
    "b": [True, True, False, None, True, False],
    "x": [-10, 2, -3, 5, -10, 20],
    "uint": [1, 2, 3, None, 1, 2],
    "enum": ["foo", "bar", "baz", None, "foo", "bar"],
    "float_vals": [1.3652, 2.7525, 3.7314, None, 1.3685, 2.7785],
    "decimal_vals": [1.3652, 2.7525, 3.7314, None, 1.3685, 2.7785],
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
    "d": [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 1),
        date(2024, 1, 2),
    ],
    "dt": [
        datetime(2024, 1, 1, 10, 30, 15, 123_456),
        datetime(2024, 1, 2, 11, 45, 30, 1),
        datetime(2024, 1, 3, 23, 59, 59, 999_001),
        datetime(2024, 1, 4, 0, 0, 0, 0),
        datetime(2024, 1, 1, 10, 30, 15, 123_456),
        datetime(2024, 1, 2, 11, 45, 30, 1),
    ],
    "binary": [b"foo", b"bar", b"baz", None, b"foo", b"bar"],
    "time": [
        time(10, 30, 15, 123_456),
        time(11, 45, 30, 1),
        time(23, 59, 59, 999_001),
        time(0, 0, 0, 0),
        time(10, 30, 15, 123_456),
        time(11, 45, 30, 1),
    ],
}
_SCHEMA = {
    "uint": pl.UInt16(),
    "decimal_vals": pl.Decimal(10, 4),
    "binary": pl.Binary(),
}
_DF = nw.from_native(
    pl.DataFrame(_DATA, schema_overrides=_SCHEMA).pipe(duckdb.from_arrow)
)


def sample_df() -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    return _DF


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


def assert_lf_eq_pl(pql_lf: pql.LazyFrame, polars_lf: pl.LazyFrame) -> None:
    assert_frame_equal(
        pql_lf.collect(), polars_lf.collect(), check_dtypes=False, check_row_order=False
    )


def on_simple_fn(pql_expr: object, pl_expr: object, fn_name: str) -> None:
    assert_eq_pl(getattr(pql_expr, fn_name)(), getattr(pl_expr, fn_name)())
