from collections.abc import Iterable

import duckdb
import polars as pl
from polars.testing import assert_frame_equal

import pql


def sample_df() -> pl.LazyFrame:
    """Create a sample DataFrame with string data for testing."""
    return pl.LazyFrame(
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


def assert_eq(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: pl.Expr | Iterable[pl.Expr]
) -> None:
    assert_frame_equal(
        sample_df().lazy().select(polars_exprs).collect(),
        pql.LazyFrame(sample_df()).select(pql_exprs).collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def test_rand() -> None:
    assert_eq((True & pql.col("a")).alias("r"), (True & pl.col("a")).alias("r"))


def test_ror() -> None:
    assert_eq((False | pql.col("a")).alias("r"), (False | pl.col("a")).alias("r"))


def test_hash() -> None:
    assert hash(pql.col("x")) == hash(pql.col("x"))


def test_repeat_by() -> None:
    assert_eq(
        pql.col("x").repeat_by(pql.col("n")).alias("repeated"),
        (pl.col("x").repeat_by(pl.col("n")).alias("repeated")),
    )
    assert_eq(
        pql.col("x").repeat_by(2).alias("repeated"),
        pl.col("x").repeat_by(2).alias("repeated"),
    )


def test_mul() -> None:
    assert_eq((pql.col("x") * 5).alias("r"), (pl.col("x") * 5).alias("r"))


def test_truediv() -> None:
    assert_eq((pql.col("x") / 5).alias("r"), (pl.col("x") / 5).alias("r"))


def test_replace() -> None:
    assert_eq(
        pql.col("x").replace(2, 99).alias("rep"),
        pl.col("x").replace(2, 99).alias("rep"),
    )


def test_all() -> None:
    assert_eq(pql.all(), pl.all())


def test_repr() -> None:
    assert "Expr" in repr(pql.col("name"))


def test_and() -> None:
    assert_eq(
        (pql.col("a") & pql.col("b")).alias("r"), (pl.col("a") & pl.col("b")).alias("r")
    )


def test_or() -> None:
    assert_eq(
        (pql.col("a") | pql.col("b")).alias("r"), (pl.col("a") | pl.col("b")).alias("r")
    )


def test_not() -> None:
    assert_eq((~pql.col("a")).alias("r"), (~pl.col("a")).alias("r"))


def test_radd() -> None:
    assert_eq((10 + pql.col("x")).alias("r"), (10 + pl.col("x")).alias("r"))


def test_rmul() -> None:
    assert_eq((10 * pql.col("x")).alias("r"), (10 * pl.col("x")).alias("r"))


def test_rtruediv() -> None:
    assert_eq((10 / pql.col("x")).alias("r"), (10 / pl.col("x")).alias("r"))


def test_eq() -> None:
    assert_eq((pql.col("x") == 2).alias("x"), (pl.col("x") == 2))


def test_lt() -> None:
    assert_eq((pql.col("x") < 3).alias("x"), (pl.col("x") < 3))


def test_gt() -> None:
    assert_eq((pql.col("x") > 2).alias("x"), (pl.col("x") > 2))


def test_ge() -> None:
    assert_eq((pql.col("x") >= 3).alias("x"), (pl.col("x") >= 3))


def test_rsub() -> None:
    assert_eq((10 - pql.col("x")).alias("x"), (10 - pl.col("x")).alias("x"))


def test_rfloordiv() -> None:
    assert_eq((10 // pql.col("x")).alias("x"), (10 // pl.col("x")).alias("x"))


def test_rmod() -> None:
    assert_eq((10 % pql.col("x")).alias("x"), (10 % pl.col("x")).alias("x"))


def test_rpow() -> None:
    assert_eq((2 ** pql.col("x")).alias("x"), (2 ** pl.col("x")).alias("x"))


def test_neg() -> None:
    assert_eq((-pql.col("x")).alias("x"), (-pl.col("x")).alias("x"))


def test_ne() -> None:
    assert_eq((pql.col("x") != 2).alias("x"), (pl.col("x") != 2).alias("x"))


def test_le() -> None:
    assert_eq((pql.col("x") <= 2).alias("x"), (pl.col("x") <= 2).alias("x"))


def test_sub() -> None:
    assert_eq((pql.col("x") - 5).alias("x"), (pl.col("x") - 5).alias("x"))


def test_floordiv() -> None:
    assert_eq((pql.col("x") // 3).alias("x"), (pl.col("x") // 3).alias("x"))


def test_is_first_distinct() -> None:
    result = pql.col("a").is_first_distinct().alias("is_first")
    expected = pl.col("a").is_first_distinct().alias("is_first")
    assert_eq(result, expected)


def test_is_last_distinct() -> None:
    result = pql.col("x").is_last_distinct().alias("is_last")
    expected = pl.col("x").is_last_distinct().alias("is_last")
    assert_eq(result, expected)


def test_sinh() -> None:
    result = pql.col("x").sinh().alias("sinh")
    expected = pl.col("x").sinh().alias("sinh")
    assert_eq(result, expected)


def test_cosh() -> None:
    result = pql.col("x").cosh().alias("cosh")
    expected = pl.col("x").cosh().alias("cosh")
    assert_eq(result, expected)


def test_tanh() -> None:
    result = pql.col("x").tanh().alias("tanh")
    expected = pl.col("x").tanh().alias("tanh")
    assert_eq(result, expected)


def test_arithmetic_operators() -> None:
    assert_eq(
        (pql.col("x").neg().alias("neg")),
        ((-pl.col("x")).alias("neg")),
    )


def test_col_getattr() -> None:
    assert_eq(pql.col.a, pl.col.a)


def test_round_modes() -> None:
    assert_eq(
        (pql.col("x").round(0, mode="half_to_even").alias("rounded")),
        (pl.col("x").round(0).alias("rounded")),
    )
    assert_eq(
        (pql.col("x").round(0, mode="round").alias("rounded")),
        (pl.col("x").round_sig_figs(1).alias("rounded")),
    )


def test_pipe() -> None:
    assert_eq(
        pql.col("x").pipe(lambda x: x * 2).alias("x"), pl.col("x").pipe(lambda x: x * 2)
    )


def test_forward_fill() -> None:
    result = pql.col("a").forward_fill().alias("forward_filled")
    expected = pl.col("a").forward_fill().alias("forward_filled")
    assert_eq(result, expected)


def test_backward_fill() -> None:
    assert_eq(
        pql.col("a").backward_fill().alias("backward_filled"),
        pl.col("a").backward_fill().alias("backward_filled"),
    )


def test_interpolate() -> None:
    assert_eq(
        pql.col("x").interpolate().alias("interpolated"),
        pl.col("x").interpolate().alias("interpolated"),
    )


def test_is_nan() -> None:
    assert_eq(
        pql.col("x").is_nan().alias("is_nan"), pl.col("x").is_nan().alias("is_nan")
    )


def test_is_null() -> None:
    assert_eq(
        pql.col("x").is_null().alias("is_null"), pl.col("x").is_null().alias("is_null")
    )


def test_is_not_null() -> None:
    result = pql.col("x").is_not_null().alias("is_not_null")
    expected = pl.col("x").is_not_null().alias("is_not_null")
    assert_eq(result, expected)


def test_is_not_nan() -> None:
    result = pql.col("x").is_not_nan().alias("is_not_nan")
    expected = pl.col("x").is_not_nan().alias("is_not_nan")
    assert_eq(result, expected)


def test_is_finite() -> None:
    result = pql.col("x").is_finite().alias("is_finite")
    expected = pl.col("x").is_finite().alias("is_finite")
    assert_eq(result, expected)


def test_is_infinite() -> None:
    result = pql.col("x").is_infinite().alias("is_infinite")
    expected = pl.col("x").is_infinite().alias("is_infinite")
    assert_eq(result, expected)


def test_fill_nan() -> None:
    result = pql.col("x").fill_nan(0.0).alias("filled")
    expected = pl.col("x").fill_nan(0.0).alias("filled")
    assert_eq(result, expected)


def test_is_duplicated() -> None:
    result = pql.col("a").is_duplicated().alias("is_dup")
    expected = pl.col("a").is_duplicated().alias("is_dup")
    assert_eq(result, expected)


def test_floor() -> None:
    result = pql.col("x").floor().alias("x_floor")
    expected = pl.col("x").floor().alias("x_floor")
    assert_eq(result, expected)


def test_ceil() -> None:
    result = pql.col("x").ceil().alias("x_ceil")
    expected = pl.col("x").ceil().alias("x_ceil")
    assert_eq(result, expected)


def test_round() -> None:
    result = pql.col("x").round(2).alias("x_round")
    expected = pl.col("x").round(2).alias("x_round")
    assert_eq(result, expected)


def test_sqrt() -> None:
    result = pql.col("x").sqrt().alias("x_sqrt")
    expected = pl.col("x").sqrt().alias("x_sqrt")
    assert_eq(result, expected)


def test_cbrt() -> None:
    result = pql.col("x").cbrt().alias("x_cbrt")
    expected = pl.col("x").cbrt().alias("x_cbrt")
    assert_eq(result, expected)


def test_log() -> None:
    result = pql.col("x").log(10).alias("x_log10")
    expected = pl.col("x").log(10).alias("x_log10")
    assert_eq(result, expected)


def test_log10() -> None:
    result = pql.col("x").log10().alias("x_log10")
    expected = pl.col("x").log10().alias("x_log10")
    assert_eq(result, expected)


def test_log1p() -> None:
    result = pql.col("x").log1p().alias("x_log1p")
    expected = pl.col("x").log1p().alias("x_log1p")
    assert_eq(result, expected)


def test_exp() -> None:
    result = pql.col("x").exp().alias("x_exp")
    expected = pl.col("x").exp().alias("x_exp")
    assert_eq(result, expected)


def test_sin() -> None:
    result = pql.col("x").sin().alias("sin_x")
    expected = pl.col("x").sin().alias("sin_x")
    assert_eq(result, expected)


def test_cos() -> None:
    result = pql.col("x").cos().alias("cos_x")
    expected = pl.col("x").cos().alias("cos_x")
    assert_eq(result, expected)


def test_tan() -> None:
    result = pql.col("x").tan().alias("tan_x")
    expected = pl.col("x").tan().alias("tan_x")
    assert_eq(result, expected)


def test_arctan() -> None:
    result = pql.col("x").arctan().alias("arctan_x")
    expected = pl.col("x").arctan().alias("arctan_x")
    assert_eq(result, expected)


def test_degrees() -> None:
    assert_eq(pql.col("x").degrees().alias("x"), pl.col("x").degrees().alias("x"))


def test_radians() -> None:
    assert_eq(pql.col("x").radians().alias("x"), pl.col("x").radians().alias("x"))


def test_sign() -> None:
    assert_eq(pql.col("x").sign().alias("sign_x"), pl.col("x").sign().alias("sign_x"))


def test_pow() -> None:
    result = (
        pql.col("x").pow(2).alias("x_squared"),
        (pql.col("x") ** 2).alias("x_squared_bis"),
    )
    expected = (
        pl.col("x").pow(2).alias("x_squared"),
        (pl.col("x") ** 2).alias("x_squared_bis"),
    )
    assert_eq(result, expected)


def test_cast() -> None:
    assert_eq(
        pql.col("x").cast(pql.String).alias("x_str"),
        pl.col("x").cast(pl.String).alias("x_str"),
    )
    assert_eq(
        pql.col("age").cast(pql.Float64).alias("age"),
        pl.col("age").cast(pl.Float64).alias("age"),
    )


def test_add() -> None:
    assert_eq(
        (
            pql.col("age").add(10).alias("age_plus_10"),
            (pql.col("age") + 10).alias("age_plus_10_bis"),
            pql.col("age").add(duckdb.ColumnExpression("x")).alias("age_plus_x"),
        ),
        (
            pl.col("age").add(10).alias("age_plus_10"),
            (pl.col("age") + 10).alias("age_plus_10_bis"),
            pl.col("age").add(pl.col("x")).alias("age_plus_x"),
        ),
    )


def test_multiply() -> None:
    assert_eq(
        (pql.col("salary").mul(2).alias("double_salary")),
        (pl.col("salary").mul(2).alias("double_salary")),
    )


def test_divide() -> None:
    assert_eq(
        (pql.col("salary").truediv(1000).alias("salary_k"),),
        (pl.col("salary").truediv(1000).alias("salary_k"),),
    )


def test_mod() -> None:
    assert_eq(
        (
            (pql.col("age") % 10).alias("age_mod_10_bis"),
            pql.col("age").mod(10).alias("age_mod_10"),
        ),
        (
            (pl.col("age") % 10).alias("age_mod_10_bis"),
            pl.col("age").mod(10).alias("age_mod_10"),
        ),
    )


def test_is_in() -> None:
    assert_eq(
        pql.col("x").is_in([2, 3]).alias("x_in"),
        pl.col("x").is_in([2, 3]).alias("x_in"),
    )


def test_abs() -> None:
    assert_eq(pql.col("x").abs().alias("x"), pl.col("x").abs().alias("x"))
