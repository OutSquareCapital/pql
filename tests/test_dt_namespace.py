import polars as pl
import pytest
from polars.expr.datetime import ExprDateTimeNameSpace

import pql
import pql._typing
from pql._expr import ExprDateTimeNameSpace as PqlExprDateTimeNameSpace

from ._utils import assert_eq_pl, on_simple_fn


def _pql_dt() -> PqlExprDateTimeNameSpace:
    return pql.col("dt").dt


def _pl_dt() -> ExprDateTimeNameSpace:
    return pl.col("dt").dt


_SIMPLE_FNS = {
    "microsecond",
    "nanosecond",
    "millisecond",
    "second",
    "minute",
    "hour",
    "day",
    "weekday",
    "ordinal_day",
    "week",
    "month",
    "quarter",
    "year",
    "iso_year",
    "century",
    "millennium",
    "date",
    "time",
    "month_start",
    "month_end",
}


@pytest.mark.parametrize("fn", sorted(_SIMPLE_FNS))
def test_simple_fns(fn: str) -> None:
    on_simple_fn(_pql_dt(), _pl_dt(), fn)


@pytest.mark.parametrize("unit", ["d", "s", "ms", "us", "ns"])
def test_epoch(unit: pql._typing.EpochTimeUnit) -> None:
    assert_eq_pl(_pql_dt().epoch(unit), _pl_dt().epoch(unit))


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_timestamp(unit: pql._typing.TimeUnit) -> None:
    assert_eq_pl(_pql_dt().timestamp(unit), _pl_dt().timestamp(unit))


def test_date_time_boundaries() -> None:
    str_fmt = "%Y-%m-%d"
    strf_fmt = "%H:%M"
    assert_eq_pl(
        (
            _pql_dt().to_string(pql.lit(str_fmt)).alias("to_string"),
            _pql_dt().strftime(pql.lit(strf_fmt)).alias("strftime"),
        ),
        (
            _pl_dt().to_string(str_fmt).alias("to_string"),
            _pl_dt().strftime(strf_fmt).alias("strftime"),
        ),
    )


def test_truncate() -> None:
    assert_eq_pl(_pql_dt().truncate("month"), _pl_dt().truncate("1mo"))


def test_round() -> None:
    assert_eq_pl(_pql_dt().round("month"), _pl_dt().round("1mo"))


def test_offset_by() -> None:
    assert_eq_pl(_pql_dt().offset_by(pql.lit("1 day")), _pl_dt().offset_by("1d"))
