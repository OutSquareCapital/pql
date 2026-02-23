from collections.abc import Iterable

import duckdb
import narwhals as nw
import polars as pl
import pyochain as pc
import pytest
from polars.testing import assert_frame_equal

import pql


def sample_df() -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    """Create a sample DataFrame with string data for testing."""
    return nw.from_native(
        duckdb.from_arrow(
            pl.DataFrame(
                {
                    "text": [
                        "  Hello World suffix  ",
                        "  foo bar baz suffix  ",
                        "  Polars is great suffix  ",
                        "  Testing string functions suffix  ",
                    ],
                    "text_nullable": [
                        "  abc  ",
                        "abc",
                        "",
                        "  ",
                    ],
                    "text_short": [
                        "a",
                        "ab",
                        "",
                        "abc",
                    ],
                    "date_str": [
                        "2024-01-15",
                        "2024-02-20",
                        "2024-03-25",
                        "2024-04-30",
                    ],
                    "dt_str": [
                        "2024-01-15 10:30:00",
                        "2024-02-20 15:45:30",
                        "2024-03-25 20:00:00",
                        "2024-04-30 23:59:59",
                    ],
                    "time_str": [
                        "10:30:00",
                        "15:45:30",
                        "20:00:00",
                        "23:59:59",
                    ],
                    "prefixed": [
                        "prefix_text",
                        "prefix_other",
                        "prefix_sample",
                        "prefix_data",
                    ],
                    "suffixed": [
                        "text_suffix",
                        "other_suffix",
                        "sample_suffix",
                        "data_suffix",
                    ],
                    "prefix_exact": [
                        "foobar",
                        "foofoobar",
                        "baab",
                        "barfoo",
                    ],
                    "suffix_exact": [
                        "foobar",
                        "foobarbar",
                        "barfoo",
                        "ababa",
                    ],
                    "prefix_col": [
                        "prefix_",
                        "prefix_",
                        "pre",
                        "data",
                    ],
                    "suffix_col": [
                        "_suffix",
                        "_suffix",
                        "suffix",
                        "data",
                    ],
                    "suffix_val": pc.Iter(range(4)).map(lambda _: "suffix").collect(),
                    "json": ['{"a": 1}', '{"a": 2}', '{"a": 3}', '{"a": 4}'],
                    "numbers": ["123.456", "456.789", "789.123", "1234.567"],
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


def test_to_uppercase() -> None:
    assert_eq(pql.col("text").str.to_uppercase(), nw.col("text").str.to_uppercase())


def test_to_lowercase() -> None:
    assert_eq(pql.col("text").str.to_lowercase(), nw.col("text").str.to_lowercase())


def test_len_chars() -> None:
    assert_eq(pql.col("text").str.len_chars(), nw.col("text").str.len_chars())


def test_contains_literal() -> None:
    assert_eq(
        pql.col("text").str.contains("lo", literal=True),
        nw.col("text").str.contains("lo", literal=True),
    )


def test_starts_with() -> None:
    assert_eq(
        pql.col("text").str.starts_with("Hello"),
        nw.col("text").str.starts_with("Hello"),
    )


def test_ends_with() -> None:
    assert_eq(
        pql.col("text").str.ends_with("suffix"), nw.col("text").str.ends_with("suffix")
    )


def test_replace() -> None:
    with pytest.raises(NotImplementedError):
        assert_eq(
            (pql.col("text").str.replace("Hello", "Hi")),
            (nw.col("text").str.replace("Hello", "Hi")),
        )
    assert_eq_pl(
        pql.col("text").str.replace("Hello", "Hi"),
        pl.col("text").str.replace("Hello", "Hi"),
    )
    assert_eq_pl(
        (
            pql.col("text").str.replace("a", "_", n=2),
            pql.col("text").str.replace("a", "_", n=0).alias("replaced_0"),
            pql.col("text").str.replace("a", "_", n=-1).alias("replaced_minus1"),
        ),
        (
            pl.col("text").str.replace("a", "_", n=2),
            pl.col("text").str.replace("a", "_", n=0).alias("replaced_0"),
            pl.col("text").str.replace("a", "_", n=-1).alias("replaced_minus1"),
        ),
    )


def test_strip_chars() -> None:
    assert_eq(pql.col("text").str.strip_chars(), nw.col("text").str.strip_chars())
    assert_eq(pql.col("text").str.strip_chars(" "), nw.col("text").str.strip_chars(" "))


def test_strip_chars_start() -> None:
    assert_eq_pl(
        (
            pql.col("text").str.strip_chars_start().alias("lstripped"),
            pql.col("text").str.strip_chars_start(" ").alias("lstripped_space"),
        ),
        (
            pl.col("text").str.strip_chars_start().alias("lstripped"),
            pl.col("text").str.strip_chars_start(" ").alias("lstripped_space"),
        ),
    )


def test_strip_chars_end() -> None:
    assert_eq_pl(
        pql.col("text").str.strip_chars_end(), pl.col("text").str.strip_chars_end()
    )
    assert_eq_pl(
        pql.col("text").str.strip_chars_end(" "),
        pl.col("text").str.strip_chars_end(" "),
    )


def test_slice() -> None:
    assert_eq(
        (
            pql.col("text").str.slice(0, 5).alias("sliced"),
            pql.col("text").str.slice(0).alias("sliced_full"),
        ),
        (
            nw.col("text").str.slice(0, 5).alias("sliced"),
            nw.col("text").str.slice(0).alias("sliced_full"),
        ),
    )
    assert_eq_pl(
        (
            pql.col("text_short").str.slice(2, 3).alias("sliced"),
            pql.col("text_short").str.slice(5).alias("sliced_full"),
        ),
        (
            pl.col("text_short").str.slice(2, 3).alias("sliced"),
            pl.col("text_short").str.slice(5).alias("sliced_full"),
        ),
    )


def test_len_bytes() -> None:
    assert_eq_pl(pql.col("text").str.len_bytes(), pl.col("text").str.len_bytes())


def test_head_str() -> None:
    assert_eq(pql.col("text").str.head(3), nw.col("text").str.head(3))


def test_tail_str() -> None:
    assert_eq(pql.col("text").str.tail(3), nw.col("text").str.tail(3))


def test_reverse_str() -> None:
    assert_eq_pl(pql.col("text").str.reverse(), pl.col("text").str.reverse())


def test_to_titlecase() -> None:
    assert_eq(pql.col("text").str.to_titlecase(), nw.col("text").str.to_titlecase())


def test_split() -> None:
    assert_eq(pql.col("text").str.split(","), nw.col("text").str.split(","))


def test_extract_all() -> None:
    assert_eq_pl(
        pql.col("text").str.extract_all(r"\d+"), pl.col("text").str.extract_all(r"\d+")
    )


def test_count_matches() -> None:
    assert_eq_pl(
        pql.col("text").str.count_matches("a", literal=True),
        pl.col("text").str.count_matches("a", literal=True),
    )


def test_strip_prefix() -> None:
    assert_eq_pl(
        pql.col("prefixed").str.strip_prefix("prefix_"),
        pl.col("prefixed").str.strip_prefix("prefix_"),
    )
    assert_eq_pl(
        pql.col("prefixed").str.strip_prefix(pql.col("prefix_col")),
        pl.col("prefixed").str.strip_prefix(pl.col("prefix_col")),
    )
    assert_eq_pl(
        pql.col("prefix_exact").str.strip_prefix("foo"),
        pl.col("prefix_exact").str.strip_prefix("foo"),
    )


def test_strip_suffix() -> None:
    assert_eq_pl(
        pql.col("suffixed").str.strip_suffix("_suffix"),
        pl.col("suffixed").str.strip_suffix("_suffix"),
    )
    assert_eq_pl(
        pql.col("suffixed").str.strip_suffix(pql.col("suffix_col")),
        pl.col("suffixed").str.strip_suffix(pl.col("suffix_col")),
    )
    assert_eq_pl(
        pql.col("suffix_exact").str.strip_suffix("bar"),
        pl.col("suffix_exact").str.strip_suffix("bar"),
    )


def test_replace_all() -> None:
    assert_eq(
        pql.col("text").str.replace_all("o", "0", literal=True),
        nw.col("text").str.replace_all("o", "0", literal=True),
    )

    assert_eq(
        pql.col("text").str.replace_all("l", "L", literal=True),
        nw.col("text").str.replace_all("l", "L", literal=True),
    )

    assert_eq(
        pql.col("text").str.replace_all(r"\d+", "X", literal=False),
        nw.col("text").str.replace_all(r"\d+", "X", literal=False),
    )
    assert_eq(
        pql.col("text").str.replace_all("suffix", pql.col("suffix_val"), literal=True),
        nw.col("text").str.replace_all("suffix", nw.col("suffix_val"), literal=True),
    )


def test_head() -> None:
    assert_eq(pql.col("text").str.head(2), nw.col("text").str.head(2))


def test_tail() -> None:
    assert_eq(pql.col("text").str.tail(2), nw.col("text").str.tail(2))


def test_contains_regex() -> None:
    assert_eq(
        pql.col("text").str.contains(r"\d+", literal=False),
        nw.col("text").str.contains(r"\d+", literal=False),
    )


def test_count_matches_literal() -> None:
    assert_eq_pl(
        (pql.col("text").str.count_matches("a", literal=True)),
        (pl.col("text").str.count_matches("a", literal=True)),
    )


def test_count_matches_regex() -> None:
    assert_eq_pl(
        (pql.col("text").str.count_matches(r"\d+", literal=False)),
        (pl.col("text").str.count_matches(r"\d+", literal=False)),
    )


def test_pad_start() -> None:
    assert_eq_pl(
        pql.col("text_short").str.pad_start(5), pl.col("text_short").str.pad_start(5)
    )
    assert_eq_pl(
        pql.col("text_short").str.pad_start(10, fill_char="*"),
        pl.col("text_short").str.pad_start(10, fill_char="*"),
    )


def test_pad_end() -> None:
    assert_eq_pl(
        pql.col("text_short").str.pad_end(5), pl.col("text_short").str.pad_end(5)
    )
    assert_eq_pl(
        pql.col("text_short").str.pad_end(10, fill_char="-"),
        pl.col("text_short").str.pad_end(10, fill_char="-"),
    )


def test_zfill() -> None:
    assert_eq_pl(pql.col("numbers").str.zfill(10), pl.col("numbers").str.zfill(10))
