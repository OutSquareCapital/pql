import narwhals as nw
import polars as pl
import pytest

import pql
from pql import _typing as t

from ._utils import assert_eq, assert_eq_pl, on_simple_fn


def test_rand() -> None:
    assert_eq((True & pql.col("a").alias("r")), (True & nw.col("a")).alias("r"))


def test_ror() -> None:
    assert_eq((False | pql.col("a").alias("r")), (False | nw.col("a")).alias("r"))


def test_hash() -> None:
    assert hash(pql.col("x")) == hash(pql.col("x"))


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


_SIMPLE_FNS = {
    "sinh",
    "cosh",
    "tanh",
    "is_finite",
    "is_infinite",
    "count",
    "len",
    "min",
    "max",
    "sum",
    "mean",
    "median",
    "mode",
    "product",
    "n_unique",
    "null_count",
    "has_nulls",
    "cot",
    "degrees",
    "radians",
    "sign",
    "floor",
    "ceil",
    "cbrt",
    "abs",
    "approx_n_unique",
    "is_last_distinct",
    "exp",
    "sin",
    "cos",
    "tan",
    "arctan",
    "arccosh",
    "arcsinh",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "diff",
}


@pytest.mark.parametrize("fn", _SIMPLE_FNS)
def test_simple_methods(fn: str) -> None:
    on_simple_fn(pql.col("x"), pl.col("x"), fn)


def test_uint_only_simple() -> None:
    assert_eq(pql.col("uint").log(2), nw.col("uint").log(2))
    assert_eq_pl(pql.col("uint").log10(), pl.col("uint").log10())
    assert_eq_pl(pql.col("uint").log1p(), pl.col("uint").log1p())
    assert_eq(pql.col("uint").sqrt(), nw.col("uint").sqrt())


def test_is_first_distinct() -> None:
    assert_eq_pl(pql.col("a").is_first_distinct(), pl.col("a").is_first_distinct())


def test_col_getattr() -> None:
    assert_eq_pl(pql.col.a, pl.col.a)


@pytest.mark.parametrize("mode", ["half_to_even", "half_away_from_zero"])
def test_round(mode: pql.sql.typing.RoundMode) -> None:
    assert_eq_pl(
        pql.col("float_vals").round(2, mode=mode),
        pl.col("float_vals").round(2, mode=mode),
    )


def test_pipe() -> None:
    assert_eq(pql.col("x").pipe(lambda x: x * 2), nw.col("x").pipe(lambda x: x * 2))


def test_forward_fill() -> None:
    assert_eq_pl(pql.col("a").forward_fill(), pl.col("a").forward_fill())


@pytest.mark.parametrize("limit", [1, 2, None])
def test_backward_fill(limit: int | None) -> None:
    assert_eq_pl(pql.col("n").backward_fill(limit), pl.col("n").backward_fill(limit))


def test_is_nan() -> None:
    assert_eq(pql.col("nan_vals").is_nan(), nw.col("nan_vals").is_nan())


def test_is_null() -> None:
    assert_eq(pql.col("n").is_null(), nw.col("n").is_null())


def test_is_not_null() -> None:
    assert_eq_pl(pql.col("n").is_not_null(), pl.col("n").is_not_null())


def test_is_not_nan() -> None:
    assert_eq_pl(pql.col("nan_vals").is_not_nan(), pl.col("nan_vals").is_not_nan())


def test_fill_nan() -> None:
    assert_eq(pql.col("nan_vals").fill_nan(0.0), nw.col("nan_vals").fill_nan(0.0))


def test_is_duplicated() -> None:
    assert_eq(pql.col("a").is_duplicated(), nw.col("a").is_duplicated())


def test_arccos() -> None:
    assert_eq_pl(
        pql.col("x").truediv(20).arccos(),
        pl.col("x").truediv(20).arccos(),
    )


def test_arcsin() -> None:
    assert_eq_pl(
        pql.col("x").truediv(20).arcsin(),
        pl.col("x").truediv(20).arcsin(),
    )


def test_arctanh() -> None:
    assert_eq_pl(
        pql.col("x").truediv(30).arctanh(),
        pl.col("x").truediv(30).arctanh(),
    )


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


@pytest.mark.parametrize("n", [0, 1, 2, -1, -2])
def test_shift(n: int) -> None:
    assert_eq_pl(pql.col("x").shift(n), pl.col("x").shift(n))


@pytest.mark.parametrize("n", [0, 1, 2, -1, -2])
def test_pct_change(n: int) -> None:
    assert_eq_pl(pql.col("x").pct_change(n), pl.col("x").pct_change(n))


@pytest.mark.parametrize("closed", ["both", "left", "right", "none"])
def test_is_between(closed: t.ClosedInterval) -> None:
    assert_eq_pl(
        pql.col("x").is_between(2, 10, closed=closed),
        pl.col("x").is_between(2, 10, closed=closed),
    )


def test_is_unique() -> None:
    assert_eq(pql.col("a").is_unique(), nw.col("a").is_unique())


@pytest.mark.parametrize("ignore_nulls", [True, False])
def test_first(ignore_nulls: bool) -> None:
    assert_eq_pl(
        pql.col("n").first(ignore_nulls=ignore_nulls),
        pl.col("n").first(ignore_nulls=ignore_nulls),
    )


def test_last() -> None:
    assert_eq_pl(pql.col("n").last(), pl.col("n").last())


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
    assert_eq_pl(pql.col("x").kurtosis(), pl.col("x").kurtosis())
    assert_eq_pl(
        pql.col("x").kurtosis(fisher=False), pl.col("x").kurtosis(fisher=False)
    )
    assert_eq_pl(pql.col("x").kurtosis(bias=False), pl.col("x").kurtosis(bias=False))


def test_skew() -> None:
    assert_eq_pl(pql.col("x").skew(), pl.col("x").skew())
    assert_eq_pl(pql.col("x").skew(bias=False), pl.col("x").skew(bias=False))


def test_quantile() -> None:
    assert_eq_pl(
        pql.col("x").quantile(0.75, interpolation=True),
        pl.col("x").quantile(0.75, "linear"),
    )
    assert_eq_pl(
        pql.col("x").quantile(0.75, interpolation=False),
        pl.col("x").quantile(0.75, "equiprobable"),
    )


@pytest.mark.parametrize("desc", [True, False])
def test_over(desc: bool) -> None:
    assert_eq_pl(
        pql.col("x").sum().over("a", order_by="n", descending=desc),
        pl.col("x").sum().over("a", order_by="n", descending=desc),
    )

    assert_eq_pl(
        pql.col("x").sum().over("a", descending=desc),
        pl.col("x").sum().over("a", descending=desc),
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


@pytest.mark.parametrize(
    "strategy", ["forward", "backward", "min", "max", "mean", "zero", "one"]
)
def test_fill_null(strategy: t.FillNullStrategy) -> None:
    assert_eq_pl(pql.col("age").fill_null(0), pl.col("age").fill_null(0))
    assert_eq_pl(
        pql.col("age").fill_null(strategy=strategy),
        pl.col("age").fill_null(strategy=strategy),
    )


@pytest.mark.parametrize("limit", [0, 1])
def test_fill_null_limit(limit: int) -> None:
    assert_eq_pl(
        pql.col("age").fill_null(strategy="forward", limit=limit),
        pl.col("age").fill_null(strategy="forward", limit=limit),
    )
    assert_eq_pl(
        pql.col("age").fill_null(strategy="backward", limit=limit),
        pl.col("age").fill_null(strategy="backward", limit=limit),
    )


def test_fill_null_no_value_or_strategy() -> None:
    msg = "must specify either a fill `value` or `strategy`"
    with pytest.raises(ValueError, match=msg):
        _ = pql.col("age").fill_null()
    with pytest.raises(ValueError, match=msg):
        _ = pl.col("age").fill_null()


def test_fill_null_limit_negative() -> None:
    with pytest.raises(
        ValueError, match="Can't process negative `limit` value for fill_null"
    ):
        _ = pql.col("age").fill_null(strategy="forward", limit=-1)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        _ = pl.col("age").fill_null(strategy="forward", limit=-1)


def test_fill_null_limit_invalid_strategy() -> None:
    err = "can only specify `limit` when strategy is set to 'backward' or 'forward'"
    with pytest.raises(ValueError, match=err):
        _ = pql.col("age").fill_null(strategy="min", limit=1)
    with pytest.raises(ValueError, match=err):
        _ = pl.col("age").fill_null(strategy="min", limit=1)
    with pytest.raises(ValueError, match=err):
        _ = pql.col("age").fill_null(0, limit=1)
    with pytest.raises(ValueError, match=err):
        _ = pl.col("age").fill_null(0, limit=1)


def test_fill_val_and_strat() -> None:
    err = "cannot specify both `value` and `strategy`"
    with pytest.raises(ValueError, match=err):
        _ = pql.col("age").fill_null(value=0, strategy="min")
    with pytest.raises(ValueError, match=err):
        _ = pl.col("age").fill_null(value=0, strategy="min")


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


def test_null_count() -> None:
    assert_eq_pl(pql.col("age").null_count(), pl.col("age").null_count())


def test_has_nulls() -> None:
    assert_eq_pl(pql.col("age").has_nulls(), pl.col("age").has_nulls())


@pytest.mark.parametrize("method", ["average", "min", "max", "dense", "ordinal"])
def test_rank(method: t.RankMethod) -> None:
    assert_eq_pl(pql.col("x").rank(method), pl.col("x").rank(method))
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
