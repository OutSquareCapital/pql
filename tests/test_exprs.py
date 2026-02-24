from collections.abc import Iterable

import duckdb
import narwhals as nw
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pql


def sample_df() -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    nan = float("nan")
    return nw.from_native(
        duckdb.from_arrow(
            pl.DataFrame(
                {
                    "a": [True, False, True, None, True, False],
                    "b": [True, True, False, None, True, False],
                    "x": [10, 2, 3, 5, 10, 20],
                    "n": [None, 3, 1, None, 2, 3],
                    "s": ["1", "2", "3", None, "1", "2"],
                    "age": [25, 30, 35, None, 25, 30],
                    "salary": [50000.0, 60000.0, 70000.0, None, 50000.0, 60000.0],
                    "nested": [[1, 2], [3, 4], [5], None, [1, 2], [3, 4]],
                    "nan_vals": [1.0, nan, 3.0, nan, 5.0, nan],
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


def test_rand() -> None:
    assert_eq((True & pql.col("a").alias("r")), (True & nw.col("a")).alias("r"))


def test_ror() -> None:
    assert_eq((False | pql.col("a").alias("r")), (False | nw.col("a")).alias("r"))


def test_hash() -> None:
    assert hash(pql.col("x")) == hash(pql.col("x"))


def test_bitwise_and() -> None:
    assert_eq_pl(pql.col("x").bitwise_and(), pl.col("x").bitwise_and())


def test_bitwise_or() -> None:
    assert_eq_pl(pql.col("x").bitwise_or(), pl.col("x").bitwise_or())


def test_bitwise_xor() -> None:
    assert_eq_pl(pql.col("x").bitwise_xor(), pl.col("x").bitwise_xor())


def test_xor() -> None:
    assert_eq_pl(pql.col("x").xor(3), pl.col("x").xor(3))
    assert_eq_pl(pql.col("x").__xor__(3), pl.col("x").xor(3))
    assert_eq_pl(
        pql.col("x").xor(pql.col("n")),
        pl.col("x").xor(pl.col("n")),
    )


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
    assert_eq(pql.col("x").mul(5), nw.col("x").__mul__(5))
    assert_eq(
        pql.col("salary").mul(2).alias("double_salary"),
        nw.col("salary").__mul__(2).alias("double_salary"),
    )


def test_truediv() -> None:
    assert_eq((pql.col("x") / 5), (nw.col("x") / 5))

    assert_eq(
        (pql.col("salary").truediv(1000).alias("salary_k"),),
        (nw.col("salary").__truediv__(1000).alias("salary_k"),),
    )


def test_replace() -> None:
    assert_eq_pl(
        pql.col("x").replace(2, 99).alias("rep"),
        pl.col("x").replace(2, 99).alias("rep"),
    )


def test_all_fn() -> None:
    assert_eq(pql.all(), nw.all())


def test_repr() -> None:
    assert "Expr" in repr(pql.col("name"))


def test_and() -> None:
    assert_eq((pql.col("a") & pql.col("b")), (nw.col("a") & nw.col("b")))


def test_or() -> None:
    assert_eq((pql.col("a") | pql.col("b")), (nw.col("a") | nw.col("b")))


def test_not() -> None:
    assert_eq((~pql.col("a")), (~nw.col("a")))


def test_radd() -> None:
    assert_eq((10 + pql.col("x").alias("r")), (10 + nw.col("x")).alias("r"))


def test_rmul() -> None:
    assert_eq((10 * pql.col("x").alias("r")), (10 * nw.col("x")).alias("r"))


def test_rtruediv() -> None:
    assert_eq((10 / pql.col("x").alias("r")), (10 / nw.col("x")).alias("r"))


def test_eq() -> None:
    assert_eq((pql.col("x") == 2), (nw.col("x") == 2))


def test_lt() -> None:
    assert_eq((pql.col("x") < 3), (nw.col("x") < 3))


def test_gt() -> None:
    assert_eq((pql.col("x") > 2), (nw.col("x") > 2))


def test_ge() -> None:
    assert_eq((pql.col("x") >= 3), (nw.col("x") >= 3))


def test_rsub() -> None:
    assert_eq((10 - pql.col("x")).alias("r"), (10 - nw.col("x")).alias("r"))


def test_rfloordiv() -> None:
    assert_eq((10 // pql.col("x")).alias("r"), (10 // nw.col("x")).alias("r"))


def test_rmod() -> None:
    assert_eq((10 % pql.col("x")).alias("r"), (10 % nw.col("x")).alias("r"))


def test_rpow() -> None:
    assert_eq((2 ** pql.col("x")).alias("r"), (2 ** nw.col("x")).alias("r"))


def test_neg() -> None:
    assert_eq_pl((-pql.col("x")), (-pl.col("x")))
    assert_eq_pl(
        pql.col("x").neg().alias("neg"),
        pl.col("x").neg().alias("neg"),
    )


def test_ne() -> None:
    assert_eq((pql.col("x") != 2), (nw.col("x") != 2))


def test_le() -> None:
    assert_eq((pql.col("x") <= 2), (nw.col("x") <= 2))


def test_sub() -> None:
    assert_eq((pql.col("x") - 5), (nw.col("x") - 5))


def test_floordiv() -> None:
    assert_eq((pql.col("x") // 3), (nw.col("x") // 3))


def test_is_first_distinct() -> None:
    assert_eq_pl(pql.col("a").is_first_distinct(), pl.col("a").is_first_distinct())


def test_is_last_distinct() -> None:
    assert_eq_pl(pql.col("x").is_last_distinct(), pl.col("x").is_last_distinct())


def test_sinh() -> None:
    assert_eq_pl(pql.col("x").sinh(), pl.col("x").sinh())


def test_cosh() -> None:
    assert_eq_pl(pql.col("x").cosh(), pl.col("x").cosh())


def test_tanh() -> None:
    assert_eq_pl(pql.col("x").tanh(), pl.col("x").tanh())


def test_col_getattr() -> None:
    assert_eq_pl(pql.col.a, pl.col.a)


def test_round_modes() -> None:
    assert_eq(pql.col("x").round(2), nw.col("x").round(2))
    assert_eq_pl(
        pql.col("x").round(2, mode="half_to_even"),
        pl.col("x").round(2, mode="half_to_even"),
    )
    assert_eq_pl(
        pql.col("x").round(2, mode="half_away_from_zero"),
        pl.col("x").round(2, mode="half_away_from_zero"),
    )


def test_pipe() -> None:
    assert_eq(pql.col("x").pipe(lambda x: x * 2), nw.col("x").pipe(lambda x: x * 2))


def test_forward_fill() -> None:
    assert_eq_pl(pql.col("a").forward_fill(), pl.col("a").forward_fill())


def test_backward_fill() -> None:
    assert_eq_pl(pql.col("n").backward_fill(), pl.col("n").backward_fill())
    assert_eq_pl(pql.col("n").backward_fill(2), pl.col("n").backward_fill(2))


def test_is_nan() -> None:
    assert_eq(pql.col("nan_vals").is_nan(), nw.col("nan_vals").is_nan())


def test_is_null() -> None:
    assert_eq(pql.col("n").is_null(), nw.col("n").is_null())


def test_is_not_null() -> None:
    assert_eq_pl(pql.col("n").is_not_null(), pl.col("n").is_not_null())


def test_is_not_nan() -> None:
    assert_eq_pl(pql.col("nan_vals").is_not_nan(), pl.col("nan_vals").is_not_nan())


def test_is_finite() -> None:
    assert_eq(pql.col("x").is_finite(), nw.col("x").is_finite())


def test_is_infinite() -> None:
    assert_eq_pl(pql.col("x").is_infinite(), pl.col("x").is_infinite())


def test_fill_nan() -> None:
    assert_eq(pql.col("nan_vals").fill_nan(0.0), nw.col("nan_vals").fill_nan(0.0))


def test_is_duplicated() -> None:
    assert_eq(pql.col("a").is_duplicated(), nw.col("a").is_duplicated())


def test_floor() -> None:
    assert_eq(pql.col("x").floor(), nw.col("x").floor())


def test_ceil() -> None:
    assert_eq(pql.col("x").ceil(), nw.col("x").ceil())


def test_round() -> None:
    assert_eq(pql.col("x").round(2), nw.col("x").round(2))


def test_sqrt() -> None:
    assert_eq(pql.col("x").sqrt(), nw.col("x").sqrt())


def test_cbrt() -> None:
    assert_eq_pl(pql.col("x").cbrt(), pl.col("x").cbrt())


def test_log() -> None:
    assert_eq(pql.col("x").log(10), nw.col("x").log(10))


def test_log10() -> None:
    assert_eq_pl(pql.col("x").log10(), pl.col("x").log10())


def test_log1p() -> None:
    assert_eq_pl(pql.col("x").log1p(), pl.col("x").log1p())


def test_exp() -> None:
    assert_eq(pql.col("x").exp(), nw.col("x").exp())


def test_sin() -> None:
    assert_eq(pql.col("x").sin(), nw.col("x").sin())


def test_cos() -> None:
    assert_eq_pl(pql.col("x").cos(), pl.col("x").cos())


def test_tan() -> None:
    assert_eq_pl(pql.col("x").tan(), pl.col("x").tan())


def test_arctan() -> None:
    assert_eq_pl(pql.col("x").arctan(), pl.col("x").arctan())


def test_arccos() -> None:
    assert_eq_pl(
        pql.col("x").truediv(20).arccos(),
        pl.col("x").truediv(20).arccos(),
    )


def test_arccosh() -> None:
    assert_eq_pl(pql.col("x").arccosh(), pl.col("x").arccosh())


def test_arcsin() -> None:
    assert_eq_pl(
        pql.col("x").truediv(20).arcsin(),
        pl.col("x").truediv(20).arcsin(),
    )


def test_arcsinh() -> None:
    assert_eq_pl(pql.col("x").arcsinh(), pl.col("x").arcsinh())


def test_arctanh() -> None:
    assert_eq_pl(
        pql.col("x").truediv(30).arctanh(),
        pl.col("x").truediv(30).arctanh(),
    )


def test_cot() -> None:
    assert_eq_pl(pql.col("x").cot(), pl.col("x").cot())


def test_degrees() -> None:
    assert_eq_pl(pql.col("x").degrees(), pl.col("x").degrees())


def test_radians() -> None:
    assert_eq_pl(pql.col("x").radians(), pl.col("x").radians())


def test_sign() -> None:
    assert_eq_pl(pql.col("x").sign(), pl.col("x").sign())


def test_pow() -> None:
    assert_eq(pql.col("x").pow(2), nw.col("x").__pow__(2))
    assert_eq(pql.col("x").__pow__(2), nw.col("x").__pow__(2))


def test_add() -> None:
    assert_eq(pql.col("age").add(10), nw.col("age").__add__(10))
    assert_eq(pql.col("age").__add__(10), nw.col("age").__add__(10))


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
    assert_eq(pql.col("x").is_in([2, 3]), nw.col("x").is_in([2, 3]))


def test_abs() -> None:
    assert_eq(pql.col("x").abs(), nw.col("x").abs())


def test_shift() -> None:
    assert_eq_pl(pql.col("x").shift(), pl.col("x").shift())
    assert_eq_pl(pql.col("x").shift(-1), pl.col("x").shift(-1))
    assert_eq_pl(pql.col("x").shift(0), pl.col("x").shift(0))


def test_diff() -> None:
    assert_eq_pl(
        pql.col("x").diff().alias("x_diff"), pl.col("x").diff().alias("x_diff")
    )


def test_pct_change() -> None:
    assert_eq_pl(pql.col("x").pct_change(), pl.col("x").pct_change())
    assert_eq_pl(pql.col("x").pct_change(2), pl.col("x").pct_change(2))
    assert_eq_pl(pql.col("x").pct_change(-1), pl.col("x").pct_change(-1))


def test_is_between() -> None:
    assert_eq_pl(
        [
            pql.col("x").is_between(2, 10).alias("x_between"),
            pql.col("x").is_between(2, 10, closed="left").alias("x_between_left"),
            pql.col("x").is_between(2, 10, closed="right").alias("x_between_right"),
            pql.col("x").is_between(2, 10, closed="none").alias("x_between_none"),
        ],
        [
            pl.col("x").is_between(2, 10).alias("x_between"),
            pl.col("x").is_between(2, 10, closed="left").alias("x_between_left"),
            pl.col("x").is_between(2, 10, closed="right").alias("x_between_right"),
            pl.col("x").is_between(2, 10, closed="none").alias("x_between_none"),
        ],
    )


def test_is_unique() -> None:
    assert_eq(pql.col("a").is_unique(), nw.col("a").is_unique())


def test_when_then_simple() -> None:
    assert_eq_pl(
        pql.when(pql.col("x").eq(5))
        .then(pql.lit("equal_to_5"))
        .otherwise(pql.lit("not_equal_to_5"))
        .alias("bucket"),
        pl.when(pl.col("x").eq(5))
        .then(pl.lit("equal_to_5"))
        .otherwise(pl.lit("not_equal_to_5"))
        .alias("bucket"),
    )


def test_when_then_chained() -> None:
    assert_eq_pl(
        pql.when(pql.col("x") > 5)
        .then(pql.lit("high"))
        .when(pql.col("x") < 5)
        .then(pql.lit("low"))
        .when(pql.col("x") == 5)
        .then(pql.lit("equal"))
        .otherwise(pql.lit("mid"))
        .alias("bucket"),
        pl.when(pl.col("x") > 5)
        .then(pl.lit("high"))
        .when(pl.col("x") < 5)
        .then(pl.lit("low"))
        .when(pl.col("x") == 5)
        .then(pl.lit("equal"))
        .otherwise(pl.lit("mid"))
        .alias("bucket"),
    )


def test_count() -> None:
    assert_eq(pql.col("x").count(), nw.col("x").count())


def test_len() -> None:
    assert_eq_pl(pql.col("x").len(), pl.col("x").len())


def test_sum() -> None:
    assert_eq_pl(pql.col("x").sum(), pl.col("x").sum())


def test_mean() -> None:
    assert_eq_pl(pql.col("x").mean(), pl.col("x").mean())


def test_median() -> None:
    assert_eq_pl(pql.col("x").median(), pl.col("x").median())


def test_min() -> None:
    assert_eq_pl(pql.col("x").min(), pl.col("x").min())


def test_max() -> None:
    assert_eq_pl(pql.col("x").max(), pl.col("x").max())


def test_first() -> None:
    assert_eq_pl(pql.col("x").first(), pl.col("x").first())
    assert_eq_pl(
        pql.col("n").first(ignore_nulls=False), pl.col("n").first(ignore_nulls=False)
    )
    assert_eq_pl(
        pql.col("n").first(ignore_nulls=True), pl.col("n").first(ignore_nulls=True)
    )
    assert_eq_pl(
        pql.col("n").first(ignore_nulls=False),
        pl.col("n").first(ignore_nulls=False),
    )
    assert_eq_pl(
        pql.col("n").first(ignore_nulls=True), pl.col("n").first(ignore_nulls=True)
    )


def test_last() -> None:
    assert_eq_pl(pql.col("n").last(), pl.col("n").last())


def test_mode() -> None:
    assert_eq_pl(pql.col("x").mode(), pl.col("x").mode())


def test_approx_n_unique() -> None:
    assert_eq_pl(pql.col("x").approx_n_unique(), pl.col("x").approx_n_unique())


def test_product() -> None:
    assert_eq_pl(pql.col("x").product(), pl.col("x").product())


def test_max_by() -> None:
    assert_eq_pl(pql.col("x").max_by("age"), pl.col("x").max_by("age"))
    assert_eq_pl(
        pql.col("salary").max_by(pql.col("x").neg()),
        pl.col("salary").max_by(pl.col("x").neg()),
    )


def test_min_by() -> None:
    assert_eq_pl(pql.col("x").min_by("age"), pl.col("x").min_by("age"))
    assert_eq_pl(
        pql.col("salary").min_by(pql.col("x").neg()),
        pl.col("salary").min_by(pl.col("x").neg()),
    )


def test_implode() -> None:
    assert_eq_pl(pql.col("x").implode(), pl.col("x").implode())


def test_unique() -> None:
    assert_eq_pl(pql.col("x").unique(), pl.col("x").unique())
    assert_eq_pl(
        (
            pql.col("x").unique().alias("x_unique_left"),
            pql.col("x").unique().add(3).alias("x_unique_right"),
        ),
        (
            pl.col("x").unique().alias("x_unique_left"),
            pl.col("x").unique().add(3).alias("x_unique_right"),
        ),
    )


def test_is_close() -> None:
    assert_eq_pl(
        pql.col("salary").is_close(
            pql.col("salary").add(0.001), abs_tol=0.01, rel_tol=0.0
        ),
        pl.col("salary").is_close(
            pl.col("salary").add(0.001), abs_tol=0.01, rel_tol=0.0
        ),
    )
    assert_eq_pl(
        pql.col("salary")
        .is_close(
            pql.col("salary").add(0.001), abs_tol=0.01, rel_tol=0.0, nans_equal=True
        )
        .alias("salary_close_nans_equal"),
        pl.col("salary")
        .is_close(
            pl.col("salary").add(0.001), abs_tol=0.01, rel_tol=0.0, nans_equal=True
        )
        .alias("salary_close_nans_equal"),
    )


def test_rolling_mean() -> None:
    assert_eq_pl(
        pql.col("x").rolling_mean(window_size=3, min_samples=2, center=False),
        pl.col("x").rolling_mean(window_size=3, min_samples=2, center=False),
    )
    assert_eq_pl(
        pql.col("x").rolling_mean(window_size=3, min_samples=2, center=True),
        pl.col("x").rolling_mean(window_size=3, min_samples=2, center=True),
    )


def test_rolling_sum() -> None:
    assert_eq_pl(
        pql.col("x").rolling_sum(window_size=3, min_samples=2, center=False),
        pl.col("x").rolling_sum(window_size=3, min_samples=2, center=False),
    )


def test_rolling_std() -> None:
    assert_eq_pl(
        pql.col("x").rolling_std(window_size=3, min_samples=2, center=False, ddof=1),
        pl.col("x").rolling_std(window_size=3, min_samples=2, center=False, ddof=1),
    )


def test_rolling_var() -> None:
    assert_eq_pl(
        pql.col("x").rolling_var(window_size=3, min_samples=2, center=False, ddof=1),
        pl.col("x").rolling_var(window_size=3, min_samples=2, center=False, ddof=1),
    )


def test_rolling_min() -> None:
    assert_eq_pl(
        pql.col("x").rolling_min(window_size=3, min_samples=2, center=False),
        pl.col("x").rolling_min(window_size=3, min_samples=2, center=False),
    )


def test_rolling_max() -> None:
    assert_eq_pl(
        pql.col("x").rolling_max(window_size=3, min_samples=2, center=False),
        pl.col("x").rolling_max(window_size=3, min_samples=2, center=False),
    )


def test_rolling_median() -> None:
    assert_eq_pl(
        pql.col("x").rolling_median(window_size=3, min_samples=2, center=False),
        pl.col("x").rolling_median(window_size=3, min_samples=2, center=False),
    )


def test_clip() -> None:
    assert_eq_pl(
        pql.col("x").clip(lower_bound=2, upper_bound=10).alias("x_clip"),
        pl.col("x").clip(lower_bound=2, upper_bound=10).alias("x_clip"),
    )
    assert_eq_pl(
        pql.col("x").clip(lower_bound=2).alias("x_clip_lower"),
        pl.col("x").clip(lower_bound=2).alias("x_clip_lower"),
    )
    assert_eq_pl(
        pql.col("x").clip(upper_bound=10).alias("x_clip_upper"),
        pl.col("x").clip(upper_bound=10).alias("x_clip_upper"),
    )
    assert_eq_pl(
        pql.col("x").clip().alias("x_clip_none"),
        pl.col("x").clip().alias("x_clip_none"),
    )


def test_kurtosis() -> None:
    assert_eq_pl(
        pql.col("x").kurtosis().alias("x_kurtosis"),
        pl.col("x").kurtosis().alias("x_kurtosis"),
    )
    assert_eq_pl(
        pql.col("x").kurtosis(fisher=False).alias("x_kurtosis_pearson"),
        pl.col("x").kurtosis(fisher=False).alias("x_kurtosis_pearson"),
    )
    assert_eq_pl(
        pql.col("x").kurtosis(bias=False).alias("x_kurtosis_unbiased"),
        pl.col("x").kurtosis(bias=False).alias("x_kurtosis_unbiased"),
    )


def test_skew() -> None:
    assert_eq_pl(
        pql.col("x").skew().alias("x_skew"),
        pl.col("x").skew().alias("x_skew"),
    )
    assert_eq_pl(
        pql.col("x").skew(bias=False).alias("x_skew_unbiased"),
        pl.col("x").skew(bias=False).alias("x_skew_unbiased"),
    )


def test_quantile() -> None:
    assert_eq_pl(
        pql.col("x").quantile(0.5, interpolation="lower").alias("x_quantile"),
        pl.col("x").quantile(0.5, interpolation="lower").alias("x_quantile"),
    )
    assert_eq_pl(
        pql.col("x").quantile(0.5, interpolation="linear").alias("x_quantile_linear"),
        pl.col("x").quantile(0.5, interpolation="linear").alias("x_quantile_linear"),
    )
    assert_eq_pl(
        pql.col("x")
        .quantile(0.5, interpolation="midpoint")
        .alias("x_quantile_midpoint"),
        pl.col("x")
        .quantile(0.5, interpolation="midpoint")
        .alias("x_quantile_midpoint"),
    )


def test_over() -> None:
    assert_eq_pl(pql.col("x").sum().over("a"), pl.col("x").sum().over("a"))


def test_over_order_by() -> None:
    assert_eq_pl(
        pql.col("x").sum().over("a", order_by="n"),
        pl.col("x").sum().over("a", order_by="n"),
    )


def test_over_descending() -> None:
    assert_eq_pl(
        pql.col("x").sum().over("a", descending=True),
        pl.col("x").sum().over("a", descending=True),
    )


def test_over_with_nulls_last() -> None:
    """Polars is currently bugged and does not handle nulls last correctly in window functions.

    Hence, we voluntarily don't test the ordering on a columns with nulls, otherwise it would fail.

    That being said, in isolation, it does work on `pql`.

    See:
        https://github.com/pola-rs/polars/issues/24989
    """
    assert_eq_pl(
        pql.col("n").first().over("a", order_by="x", nulls_last=True),
        pl.col("n").first().over("a", order_by="x", nulls_last=True),
    )


def test_fill_null() -> None:
    assert_eq_pl(pql.col("age").fill_null(0), pl.col("age").fill_null(0))
    assert_eq_pl(
        pql.col("age").fill_null(strategy="forward"),
        pl.col("age").fill_null(strategy="forward"),
    )
    assert_eq_pl(
        pql.col("age").fill_null(strategy="backward"),
        pl.col("age").fill_null(strategy="backward"),
    )
    assert_eq_pl(
        pql.col("age").fill_null(strategy="forward", limit=0),
        pl.col("age").fill_null(strategy="forward", limit=0),
    )
    assert_eq_pl(
        pql.col("age").fill_null(strategy="forward", limit=1),
        pl.col("age").fill_null(strategy="forward", limit=1),
    )
    assert_eq_pl(
        pql.col("age").fill_null(strategy="backward", limit=0),
        pl.col("age").fill_null(strategy="backward", limit=0),
    )
    assert_eq_pl(
        pql.col("age").fill_null(strategy="backward", limit=1),
        pl.col("age").fill_null(strategy="backward", limit=1),
    )
    assert_eq_pl(
        pql.col("age").fill_null(strategy="min"),
        pl.col("age").fill_null(strategy="min"),
    )
    assert_eq_pl(
        pql.col("age").fill_null(strategy="max"),
        pl.col("age").fill_null(strategy="max"),
    )
    assert_eq_pl(
        pql.col("age").fill_null(strategy="mean"),
        pl.col("age").fill_null(strategy="mean"),
    )
    assert_eq_pl(
        pql.col("age").fill_null(strategy="zero"),
        pl.col("age").fill_null(strategy="zero"),
    )
    assert_eq_pl(
        pql.col("age").fill_null(strategy="one"),
        pl.col("age").fill_null(strategy="one"),
    )


def test_fill_null_no_value_or_strategy() -> None:
    msg = "must specify either a fill `value` or `strategy`"
    with pytest.raises(ValueError, match=msg):
        pql.col("age").fill_null()
    with pytest.raises(ValueError, match=msg):
        pl.col("age").fill_null()


def test_fill_null_limit_negative() -> None:
    with pytest.raises(
        ValueError, match="Can't process negative `limit` value for fill_null"
    ):
        pql.col("age").fill_null(strategy="forward", limit=-1)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        pl.col("age").fill_null(strategy="forward", limit=-1)


def test_fill_null_limit_invalid_strategy() -> None:
    err = "can only specify `limit` when strategy is set to 'backward' or 'forward'"
    with pytest.raises(ValueError, match=err):
        pql.col("age").fill_null(strategy="min", limit=1)
    with pytest.raises(ValueError, match=err):
        pl.col("age").fill_null(strategy="min", limit=1)
    with pytest.raises(ValueError, match=err):
        pql.col("age").fill_null(0, limit=1)
    with pytest.raises(ValueError, match=err):
        pl.col("age").fill_null(0, limit=1)


def test_fill_val_and_strat() -> None:
    err = "cannot specify both `value` and `strategy`"
    with pytest.raises(ValueError, match=err):
        pql.col("age").fill_null(value=0, strategy="min")
    with pytest.raises(ValueError, match=err):
        pl.col("age").fill_null(value=0, strategy="min")


def test_std() -> None:
    assert_eq_pl(pql.col("x").std(), pl.col("x").std())
    assert_eq_pl(pql.col("x").std(ddof=0), pl.col("x").std(ddof=0))


def test_var() -> None:
    assert_eq_pl(pql.col("x").var(), pl.col("x").var())
    assert_eq_pl(pql.col("x").var(ddof=0), pl.col("x").var(ddof=0))


def test_all() -> None:
    assert_eq_pl(pql.col("x").gt(0).all(), pl.col("x").gt(0).all())


def test_any() -> None:
    assert_eq_pl(pql.col("x").gt(10).any(), pl.col("x").gt(10).any())


def test_n_unique() -> None:
    assert_eq_pl(pql.col("x").n_unique(), pl.col("x").n_unique())


def test_null_count() -> None:
    assert_eq_pl(pql.col("age").null_count(), pl.col("age").null_count())


def test_has_nulls() -> None:
    assert_eq_pl(pql.col("age").has_nulls(), pl.col("age").has_nulls())


def test_rank() -> None:
    assert_eq_pl(pql.col("x").rank(), pl.col("x").rank())
    assert_eq_pl(pql.col("x").rank(method="min"), pl.col("x").rank(method="min"))
    assert_eq_pl(pql.col("x").rank(method="max"), pl.col("x").rank(method="max"))
    assert_eq_pl(pql.col("x").rank(method="dense"), pl.col("x").rank(method="dense"))
    assert_eq_pl(
        pql.col("x").rank(method="ordinal"), pl.col("x").rank(method="ordinal")
    )
    assert_eq_pl(pql.col("x").rank(descending=True), pl.col("x").rank(descending=True))


def test_cum_count() -> None:
    assert_eq_pl(pql.col("x").cum_count(), pl.col("x").cum_count())
    assert_eq_pl(
        pql.col("x").cum_count(reverse=True), pl.col("x").cum_count(reverse=True)
    )


def test_cum_sum() -> None:
    assert_eq_pl(pql.col("x").cum_sum(), pl.col("x").cum_sum())
    assert_eq_pl(pql.col("x").cum_sum(reverse=True), pl.col("x").cum_sum(reverse=True))


def test_cum_prod() -> None:
    assert_eq_pl(pql.col("x").cum_prod(), pl.col("x").cum_prod())
    assert_eq_pl(
        pql.col("x").cum_prod(reverse=True), pl.col("x").cum_prod(reverse=True)
    )


def test_cum_min() -> None:
    assert_eq_pl(pql.col("x").cum_min(), pl.col("x").cum_min())
    assert_eq_pl(pql.col("x").cum_min(reverse=True), pl.col("x").cum_min(reverse=True))


def test_cum_max() -> None:
    assert_eq_pl(pql.col("x").cum_max(), pl.col("x").cum_max())
    assert_eq_pl(pql.col("x").cum_max(reverse=True), pl.col("x").cum_max(reverse=True))
