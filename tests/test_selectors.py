from __future__ import annotations

import polars as pl
import polars.selectors as cs_pl

import pql
import pql.selectors as cs

from ._utils import assert_eq_pl, assert_lf_eq_pl, sample_df

_SAMPLE_DF = sample_df().to_native().pl(lazy=True)


# ──── numeric ────


def test_numeric_select() -> None:
    assert_eq_pl(cs.numeric(), cs_pl.numeric())


def test_numeric_with_columns() -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(_SAMPLE_DF).select("s").with_columns(cs.numeric()),
        _SAMPLE_DF.select("s").with_columns(cs_pl.numeric()),
    )


def test_string_select() -> None:
    assert_eq_pl(cs.string(), cs_pl.string())


def test_boolean_select() -> None:
    assert_eq_pl(cs.boolean(), cs_pl.boolean())


def test_by_dtype_single() -> None:
    assert_eq_pl(cs.by_dtype(pql.Boolean), cs_pl.by_dtype(pl.Boolean))


def test_by_dtype_multiple() -> None:
    assert_eq_pl(
        cs.by_dtype(pql.Float64, pql.Int64), cs_pl.by_dtype(pl.Float64, pl.Int64)
    )


def test_union() -> None:
    assert_eq_pl(
        cs.numeric().union(cs.string()), cs_pl.numeric().__or__(cs_pl.string())
    )

    assert_eq_pl(
        cs.numeric().__or__(cs.string()), cs_pl.numeric().__or__(cs_pl.string())
    )

    assert_lf_eq_pl(
        pql.LazyFrame(_SAMPLE_DF).select(cs.boolean().__or__(pql.lit(value=True))),
        _SAMPLE_DF.select(cs_pl.boolean().__or__(pl.lit(value=True))),
    )


def test_intersection() -> None:
    assert_eq_pl(
        cs.numeric().intersection(cs.by_dtype(pql.Int64)),
        cs_pl.numeric().__and__(cs_pl.by_dtype(pl.Int64)),
    )

    assert_eq_pl(
        cs.numeric().__and__(cs.by_dtype(pql.Int64)),
        cs_pl.numeric().__and__(cs_pl.by_dtype(pl.Int64)),
    )

    assert_lf_eq_pl(
        pql.LazyFrame(_SAMPLE_DF).select(cs.boolean().__and__(pql.lit(value=True))),
        _SAMPLE_DF.select(cs_pl.boolean().__and__(pl.lit(value=True))),
    )


def test_difference() -> None:
    assert_eq_pl(
        cs.numeric().difference(cs.by_dtype(pql.Float64)),
        cs_pl.numeric().__sub__(cs_pl.by_dtype(pl.Float64)),
    )
    assert_eq_pl(
        cs.numeric().__sub__(cs.by_dtype(pql.Float64)),
        cs_pl.numeric().__sub__(cs_pl.by_dtype(pl.Float64)),
    )

    assert_lf_eq_pl(
        pql.LazyFrame(_SAMPLE_DF).select(cs.numeric().__sub__(pql.lit(1))),
        _SAMPLE_DF.select(cs_pl.numeric().__sub__(pl.lit(1))),
    )


def test_complement() -> None:
    assert_eq_pl(cs.boolean().complement(), cs_pl.boolean().__invert__())
    assert_eq_pl(cs.boolean().__invert__(), cs_pl.boolean().__invert__())
    assert_eq_pl(cs.numeric().complement(), cs_pl.numeric().__invert__())


def test_selector_with_suffix() -> None:
    assert_eq_pl(
        cs.boolean().name.suffix("_flag"), cs_pl.boolean().name.suffix("_flag")
    )


def test_selector_cast() -> None:
    assert_eq_pl(cs.boolean().cast(pql.Int32()), cs_pl.boolean().cast(pl.Int32))


def test_selector_in_group_by_agg() -> None:
    """We need to filter null values to avoid errors on `sum`."""
    assert_lf_eq_pl(
        pql.LazyFrame(_SAMPLE_DF)
        .filter(pql.col("a").is_not_null())
        .group_by("a")
        .agg(cs.numeric().sum())
        .sort("a"),
        _SAMPLE_DF.filter(pl.col("a").is_not_null())
        .group_by("a")
        .agg(cs_pl.numeric().sum())
        .sort("a"),
    )


def test_empty_selector() -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(_SAMPLE_DF).select(pql.col("a")).select(cs.boolean()),
        _SAMPLE_DF.select(pl.col("a")).select(cs_pl.boolean()),
    )
