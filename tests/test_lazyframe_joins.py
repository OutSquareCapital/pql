import polars as pl
import pytest

import pql
import pql._typing as t

from ._utils import assert_lf_eq_pl


@pytest.mark.parametrize("how", t.JoinStrategy.__args__)
def test_join(how: t.JoinStrategy) -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    right = pl.DataFrame({"id": [2, 3, 4], "b": [200, 300, 400]})
    assert_lf_eq_pl(
        pql.LazyFrame(left).join(pql.LazyFrame(right), on="id", how=how),
        left.lazy().join(right.lazy(), on="id", how=how),
    )


def test_join_left_on_right_on() -> None:
    left = pl.DataFrame({"lid": [1, 2, 3], "a": [10, 20, 30]})
    right = pl.DataFrame({"rid": [2, 3, 4], "b": [200, 300, 400]})
    assert_lf_eq_pl(
        pql.LazyFrame(left).join(
            pql.LazyFrame(right), left_on="lid", right_on="rid", how="inner"
        ),
        left.lazy().join(right.lazy(), left_on="lid", right_on="rid", how="inner"),
    )


def test_join_cross() -> None:
    left = pl.DataFrame({"a": [1, 2]})
    right = pl.DataFrame({"b": [10, 20]})
    assert_lf_eq_pl(
        pql.LazyFrame(left).join_cross(pql.LazyFrame(right)),
        left.lazy().join(right.lazy(), how="cross"),
    )


def test_join_cross_with_keys_error() -> None:
    left = pl.DataFrame({"id": [1, 2], "a": [10, 20]})
    right = pl.DataFrame({"id": [2, 3], "b": [200, 300]})
    with pytest.raises(ValueError, match="Can not pass"):
        _ = pql.LazyFrame(left).join_cross(pql.LazyFrame(right), on="id")


def test_join_on_multiple_keys() -> None:
    left = pl.DataFrame({"id1": [1, 2, 3], "id2": ["a", "b", "c"], "a": [10, 20, 30]})
    right = pl.DataFrame(
        {"id1": [2, 3, 4], "id2": ["b", "c", "d"], "b": [200, 300, 400]}
    )
    assert_lf_eq_pl(
        pql.LazyFrame(left).join(pql.LazyFrame(right), on=["id1", "id2"], how="inner"),
        left.lazy().join(right.lazy(), on=["id1", "id2"], how="inner"),
    )


def test_join_column_overlap() -> None:
    left = pl.DataFrame({"id": [1, 2], "a": [10, 20]})
    right = pl.DataFrame({"id": [1, 2], "a": [100, 200]})
    result = pql.LazyFrame(left).join(pql.LazyFrame(right), on="id")
    expected = left.lazy().join(right.lazy(), on="id")
    assert_lf_eq_pl(result, expected)


@pytest.mark.parametrize("strategy", t.AsofJoinStrategy.__args__)
def test_join_asof_strat(strategy: t.AsofJoinStrategy) -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "g2": ["x", "x", "y"], "b": [100, 200, 300]})
    assert_lf_eq_pl(
        pql.LazyFrame(left).join_asof(
            pql.LazyFrame(right),
            left_on="t",
            right_on="u",
            by_left="g",
            by_right="g2",
            strategy=strategy,
        ),
        left.lazy().join_asof(
            right.lazy(),
            left_on="t",
            right_on="u",
            by_left="g",
            by_right="g2",
            strategy=strategy,
        ),
    )


def test_join_asof_error_on_and_left_on() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "b": [100, 200, 300]})
    with pytest.raises(ValueError, match="If `on` is specified"):
        _ = pql.LazyFrame(left).join_asof(
            pql.LazyFrame(right), on="t", left_on="t", right_on="u"
        )


def test_join_asof_error_no_keys() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "b": [100, 200, 300]})
    with pytest.raises(ValueError, match="Either"):
        _ = pql.LazyFrame(left).join_asof(pql.LazyFrame(right))


def test_join_asof_error_left_on_without_right_on() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "b": [100, 200, 300]})
    with pytest.raises(ValueError, match="Either"):
        _ = pql.LazyFrame(left).join_asof(pql.LazyFrame(right), left_on="t")


left_asof_error = pl.LazyFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
right_asof_error = pl.LazyFrame(
    {"u": [0, 3, 8], "g2": ["x", "x", "y"], "b": [100, 200, 300]}
)


def test_join_asof_error_by_and_by_left() -> None:
    with pytest.raises(ValueError, match="If `by` is specified"):
        _ = left_asof_error.join_asof(
            right_asof_error, left_on="t", right_on="u", by="g", by_left="g"
        )


def test_join_asof_error_by_left_without_by_right() -> None:
    with pytest.raises(ValueError, match="Can not specify only"):
        _ = left_asof_error.join_asof(
            right_asof_error, left_on="t", right_on="u", by_left="g"
        )


def test_join_asof_error_unequal_by_lengths() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        _ = left_asof_error.join_asof(
            right_asof_error,
            left_on="t",
            right_on="u",
            by_left="g",
            by_right=["g2", "b"],
        )


def test_join_left_on_right_on_length_mismatch() -> None:
    left = pl.DataFrame({"id1": [1, 2], "id2": ["a", "b"], "a": [10, 20]})
    right = pl.DataFrame({"id1": [1, 2], "b": [100, 200]})
    with pytest.raises(ValueError, match="same length"):
        _ = pql.LazyFrame(left).join(
            pql.LazyFrame(right), left_on=["id1", "id2"], right_on="id1", how="inner"
        )


def test_join_left_with_multiple_keys() -> None:
    left = pl.DataFrame({"id1": [1, 2, 3], "id2": ["a", "b", "c"], "a": [10, 20, 30]})
    right = pl.DataFrame(
        {"id1": [2, 3, 4], "id2": ["b", "c", "d"], "b": [200, 300, 400]}
    )
    result = pql.LazyFrame(left).join(
        pql.LazyFrame(right),
        left_on=["id1", "id2"],
        right_on=["id1", "id2"],
        how="left",
    )
    expected = left.lazy().join(
        right.lazy(),
        left_on=["id1", "id2"],
        right_on=["id1", "id2"],
        how="left",
    )
    assert_lf_eq_pl(result, expected)
