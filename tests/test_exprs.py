from collections.abc import Iterable

import duckdb
import narwhals as nw
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pql


def sample_df() -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    """Create a sample DataFrame with string data for testing."""
    return nw.from_native(
        duckdb.from_arrow(
            pl.DataFrame(
                {
                    "a": [True, False, True, None],
                    "b": [True, True, False, None],
                    "x": [10, 2, 3, None],
                    "n": [2, 3, 1, None],
                    "s": ["1", "2", "3", None],
                    "age": [25, 30, 35, None],
                    "salary": [50000.0, 60000.0, 70000.0, None],
                    "nested": [[1, 2], [3, 4], [5], None],
                }
            )
        )
    )


def assert_eq(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: nw.Expr | Iterable[nw.Expr]
) -> None:
    assert_frame_equal(
        sample_df().lazy().select(polars_exprs).to_native().pl(),
        pql.LazyFrame(sample_df().to_native()).select(pql_exprs).collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def assert_eq_pl(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: pl.Expr | Iterable[pl.Expr]
) -> None:
    assert_frame_equal(
        sample_df().to_native().pl(lazy=True).select(polars_exprs).collect(),
        pql.LazyFrame(sample_df().to_native()).select(pql_exprs).collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def test_rand() -> None:
    assert_eq((True & pql.col("a")).alias("r"), (True & nw.col("a")).alias("r"))


def test_ror() -> None:
    assert_eq((False | pql.col("a")).alias("r"), (False | nw.col("a")).alias("r"))


def test_hash() -> None:
    assert hash(pql.col("x")) == hash(pql.col("x"))


def test_repeat_by() -> None:
    assert_eq_pl(
        pql.col("x").repeat_by(pql.col("n")).alias("repeated"),
        (pl.col("x").repeat_by(pl.col("n")).alias("repeated")),
    )
    assert_eq_pl(
        pql.col("x").repeat_by(2).alias("repeated"),
        pl.col("x").repeat_by(2).alias("repeated"),
    )


def test_mul() -> None:
    assert_eq((pql.col("x") * 5).alias("r"), (nw.col("x") * 5).alias("r"))
    assert_eq(
        (pql.col("salary").mul(2).alias("double_salary")),
        (nw.col("salary").__mul__(2).alias("double_salary")),
    )


def test_truediv() -> None:
    assert_eq((pql.col("x") / 5).alias("r"), (nw.col("x") / 5).alias("r"))

    assert_eq(
        (pql.col("salary").truediv(1000).alias("salary_k"),),
        (nw.col("salary").__truediv__(1000).alias("salary_k"),),
    )


def test_replace() -> None:
    assert_eq_pl(
        pql.col("x").replace(2, 99).alias("rep"),
        pl.col("x").replace(2, 99).alias("rep"),
    )


def test_all() -> None:
    assert_eq(pql.all(), nw.all())


def test_repr() -> None:
    assert "Expr" in repr(pql.col("name"))


def test_and() -> None:
    assert_eq(
        (pql.col("a") & pql.col("b")).alias("r"), (nw.col("a") & nw.col("b")).alias("r")
    )


def test_or() -> None:
    assert_eq(
        (pql.col("a") | pql.col("b")).alias("r"), (nw.col("a") | nw.col("b")).alias("r")
    )


def test_not() -> None:
    assert_eq((~pql.col("a")).alias("r"), (~nw.col("a")).alias("r"))


def test_radd() -> None:
    assert_eq((10 + pql.col("x")).alias("r"), (10 + nw.col("x")).alias("r"))


def test_rmul() -> None:
    assert_eq((10 * pql.col("x")).alias("r"), (10 * nw.col("x")).alias("r"))


def test_rtruediv() -> None:
    assert_eq((10 / pql.col("x")).alias("r"), (10 / nw.col("x")).alias("r"))


def test_eq() -> None:
    assert_eq((pql.col("x") == 2).alias("x"), (nw.col("x") == 2))


def test_lt() -> None:
    assert_eq((pql.col("x") < 3).alias("x"), (nw.col("x") < 3))


def test_gt() -> None:
    assert_eq((pql.col("x") > 2).alias("x"), (nw.col("x") > 2))


def test_ge() -> None:
    assert_eq((pql.col("x") >= 3).alias("x"), (nw.col("x") >= 3))


def test_rsub() -> None:
    assert_eq((10 - pql.col("x")).alias("x"), (10 - nw.col("x")).alias("x"))


def test_rfloordiv() -> None:
    assert_eq((10 // pql.col("x")).alias("x"), (10 // nw.col("x")).alias("x"))


def test_rmod() -> None:
    assert_eq((10 % pql.col("x")).alias("x"), (10 % nw.col("x")).alias("x"))


def test_rpow() -> None:
    assert_eq((2 ** pql.col("x")).alias("x"), (2 ** nw.col("x")).alias("x"))


def test_neg() -> None:
    assert_eq_pl((-pql.col("x")).alias("x"), (-pl.col("x")).alias("x"))
    assert_eq_pl(
        pql.col("x").neg().alias("neg"),
        pl.col("x").neg().alias("neg"),
    )


def test_ne() -> None:
    assert_eq((pql.col("x") != 2).alias("x"), (nw.col("x") != 2).alias("x"))


def test_le() -> None:
    assert_eq((pql.col("x") <= 2).alias("x"), (nw.col("x") <= 2).alias("x"))


def test_sub() -> None:
    assert_eq((pql.col("x") - 5).alias("x"), (nw.col("x") - 5).alias("x"))


def test_floordiv() -> None:
    assert_eq((pql.col("x") // 3).alias("x"), (nw.col("x") // 3).alias("x"))


def test_is_first_distinct() -> None:
    with pytest.raises(nw.exceptions.InvalidOperationError):
        assert_eq(
            pql.col("a").is_first_distinct().alias("is_first"),
            nw.col("a").is_first_distinct().alias("is_first"),
        )
    assert_eq_pl(
        pql.col("a").is_first_distinct().alias("is_first"),
        pl.col("a").is_first_distinct().alias("is_first"),
    )


def test_is_last_distinct() -> None:
    with pytest.raises(nw.exceptions.InvalidOperationError):
        assert_eq(
            pql.col("x").is_last_distinct().alias("is_last"),
            nw.col("x").is_last_distinct().alias("is_last"),
        )
    assert_eq_pl(
        pql.col("x").is_last_distinct().alias("is_last"),
        pl.col("x").is_last_distinct().alias("is_last"),
    )


def test_sinh() -> None:
    assert_eq_pl(pql.col("x").sinh().alias("sinh"), pl.col("x").sinh().alias("sinh"))


def test_cosh() -> None:
    assert_eq_pl(pql.col("x").cosh().alias("cosh"), pl.col("x").cosh().alias("cosh"))


def test_tanh() -> None:
    assert_eq_pl(pql.col("x").tanh().alias("tanh"), pl.col("x").tanh().alias("tanh"))


def test_col_getattr() -> None:
    assert_eq_pl(pql.col.a, pl.col.a)


def test_round_modes() -> None:
    assert_eq(
        pql.col("x").round(2).alias("rounded"),
        nw.col("x").round(2).alias("rounded"),
    )
    assert_eq_pl(
        pql.col("x").round(2, mode="half_to_even").alias("rounded"),
        pl.col("x").round(2, mode="half_to_even").alias("rounded"),
    )
    assert_eq_pl(
        pql.col("x").round(2, mode="half_away_from_zero").alias("rounded"),
        pl.col("x").round(2, mode="half_away_from_zero").alias("rounded"),
    )


def test_pipe() -> None:
    assert_eq(
        pql.col("x").pipe(lambda x: x * 2).alias("x"), nw.col("x").pipe(lambda x: x * 2)
    )


def test_forward_fill() -> None:
    assert_eq_pl(
        pql.col("a").forward_fill().alias("forward_filled"),
        pl.col("a").forward_fill().alias("forward_filled"),
    )


def test_backward_fill() -> None:
    assert_eq_pl(
        pql.col("a").backward_fill().alias("backward_filled"),
        pl.col("a").backward_fill().alias("backward_filled"),
    )


def test_interpolate() -> None:
    assert_eq_pl(
        pql.col("x").interpolate().alias("interpolated"),
        pl.col("x").interpolate().alias("interpolated"),
    )


def test_is_nan() -> None:
    assert_eq(
        pql.col("x").is_nan().alias("is_nan"), nw.col("x").is_nan().alias("is_nan")
    )


def test_is_null() -> None:
    assert_eq(
        pql.col("x").is_null().alias("is_null"), nw.col("x").is_null().alias("is_null")
    )


def test_is_not_null() -> None:
    assert_eq_pl(
        pql.col("x").is_not_null().alias("is_not_null"),
        pl.col("x").is_not_null().alias("is_not_null"),
    )


def test_is_not_nan() -> None:
    assert_eq_pl(
        pql.col("x").is_not_nan().alias("is_not_nan"),
        pl.col("x").is_not_nan().alias("is_not_nan"),
    )


def test_is_finite() -> None:
    assert_eq(
        pql.col("x").is_finite().alias("is_finite"),
        nw.col("x").is_finite().alias("is_finite"),
    )


def test_is_infinite() -> None:
    assert_eq_pl(
        pql.col("x").is_infinite().alias("is_infinite"),
        pl.col("x").is_infinite().alias("is_infinite"),
    )


def test_fill_nan() -> None:
    assert_eq(
        pql.col("x").fill_nan(0.0).alias("filled"),
        nw.col("x").fill_nan(0.0).alias("filled"),
    )


def test_is_duplicated() -> None:
    assert_eq(
        pql.col("a").is_duplicated().alias("is_dup"),
        nw.col("a").is_duplicated().alias("is_dup"),
    )


def test_floor() -> None:
    assert_eq(
        pql.col("x").floor().alias("x_floor"), nw.col("x").floor().alias("x_floor")
    )


def test_ceil() -> None:
    result = pql.col("x").ceil().alias("x_ceil")
    expected = nw.col("x").ceil().alias("x_ceil")
    assert_eq(result, expected)


def test_round() -> None:
    result = pql.col("x").round(2).alias("x_round")
    expected = nw.col("x").round(2).alias("x_round")
    assert_eq(result, expected)


def test_sqrt() -> None:
    result = pql.col("x").sqrt().alias("x_sqrt")
    expected = nw.col("x").sqrt().alias("x_sqrt")
    assert_eq(result, expected)


def test_cbrt() -> None:
    assert_eq_pl(
        pql.col("x").cbrt().alias("x_cbrt"), pl.col("x").cbrt().alias("x_cbrt")
    )


def test_log() -> None:
    result = pql.col("x").log(10).alias("x_log10")
    expected = nw.col("x").log(10).alias("x_log10")
    assert_eq(result, expected)


def test_log10() -> None:
    assert_eq_pl(
        pql.col("x").log10().alias("x_log10"), pl.col("x").log10().alias("x_log10")
    )


def test_log1p() -> None:
    assert_eq_pl(
        pql.col("x").log1p().alias("x_log1p"), pl.col("x").log1p().alias("x_log1p")
    )


def test_exp() -> None:
    result = pql.col("x").exp().alias("x_exp")
    expected = nw.col("x").exp().alias("x_exp")
    assert_eq(result, expected)


def test_sin() -> None:
    result = pql.col("x").sin().alias("sin_x")
    expected = nw.col("x").sin().alias("sin_x")
    assert_eq(result, expected)


def test_cos() -> None:
    assert_eq_pl(pql.col("x").cos().alias("cos_x"), pl.col("x").cos().alias("cos_x"))


def test_tan() -> None:
    assert_eq_pl(pql.col("x").tan().alias("tan_x"), pl.col("x").tan().alias("tan_x"))


def test_arctan() -> None:
    assert_eq_pl(
        pql.col("x").arctan().alias("arctan_x"), pl.col("x").arctan().alias("arctan_x")
    )


def test_degrees() -> None:
    assert_eq_pl(pql.col("x").degrees().alias("x"), pl.col("x").degrees().alias("x"))


def test_radians() -> None:
    assert_eq_pl(pql.col("x").radians().alias("x"), pl.col("x").radians().alias("x"))


def test_sign() -> None:
    assert_eq_pl(
        pql.col("x").sign().alias("sign_x"), pl.col("x").sign().alias("sign_x")
    )


def test_pow() -> None:
    assert_eq(
        (
            pql.col("x").pow(2).alias("x_squared"),
            (pql.col("x") ** 2).alias("x_squared_bis"),
        ),
        (
            nw.col("x").__pow__(2).alias("x_squared"),
            (nw.col("x") ** 2).alias("x_squared_bis"),
        ),
    )


def test_cast() -> None:
    assert_eq(
        pql.col("x").cast(pql.String).alias("x_str"),
        nw.col("x").cast(nw.String).alias("x_str"),
    )
    assert_eq(
        pql.col("age").cast(pql.Float64).alias("age"),
        nw.col("age").cast(nw.Float64).alias("age"),
    )


def test_add() -> None:
    assert_eq(
        (
            pql.col("age").add(10).alias("age_plus_10"),
            (pql.col("age") + 10).alias("age_plus_10_bis"),
        ),
        (
            nw.col("age").__add__(10).alias("age_plus_10"),
            (nw.col("age") + 10).alias("age_plus_10_bis"),
        ),
    )


def test_mod() -> None:
    assert_eq(
        (
            (pql.col("age") % 10).alias("age_mod_10_bis"),
            pql.col("age").mod(10).alias("age_mod_10"),
        ),
        (
            (nw.col("age") % 10).alias("age_mod_10_bis"),
            nw.col("age").__mod__(10).alias("age_mod_10"),
        ),
    )


def test_is_in() -> None:
    assert_eq(
        pql.col("x").is_in([2, 3]).alias("x_in"),
        nw.col("x").is_in([2, 3]).alias("x_in"),
    )


def test_abs() -> None:
    assert_eq(pql.col("x").abs().alias("x"), nw.col("x").abs().alias("x"))
