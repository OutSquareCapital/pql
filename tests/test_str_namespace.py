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


def test_to_uppercase() -> None:
    assert_eq(
        (pql.col("text").str.to_uppercase().alias("upper")),
        (nw.col("text").str.to_uppercase().alias("upper")),
    )


def test_to_lowercase() -> None:
    assert_eq(
        (pql.col("text").str.to_lowercase().alias("lower")),
        (nw.col("text").str.to_lowercase().alias("lower")),
    )


def test_len_chars() -> None:
    assert_eq(
        (pql.col("text").str.len_chars().alias("length")),
        (nw.col("text").str.len_chars().alias("length")),
    )


def test_contains_literal() -> None:
    assert_eq(
        (pql.col("text").str.contains("lo", literal=True).alias("contains_lo")),
        (nw.col("text").str.contains("lo", literal=True).alias("contains_lo")),
    )


def test_starts_with() -> None:
    assert_eq(
        (pql.col("text").str.starts_with("Hello").alias("starts_hello")),
        (nw.col("text").str.starts_with("Hello").alias("starts_hello")),
    )


def test_ends_with() -> None:
    assert_eq(
        (pql.col("text").str.ends_with("suffix").alias("ends_suffix")),
        (nw.col("text").str.ends_with("suffix").alias("ends_suffix")),
    )


def test_replace() -> None:
    with pytest.raises(NotImplementedError):
        assert_eq(
            (pql.col("text").str.replace("Hello", "Hi").alias("replaced")),
            (nw.col("text").str.replace("Hello", "Hi").alias("replaced")),
        )
    assert_eq_pl(
        (pql.col("text").str.replace("Hello", "Hi").alias("replaced")),
        (pl.col("text").str.replace("Hello", "Hi").alias("replaced")),
    )
    assert_eq_pl(
        (
            pql.col("text").str.replace("a", "_", n=2).alias("replaced"),
            pql.col("text").str.replace("a", "_", n=0).alias("replaced_0"),
            pql.col("text").str.replace("a", "_", n=-1).alias("replaced_minus1"),
        ),
        (
            pl.col("text").str.replace("a", "_", n=2).alias("replaced"),
            pl.col("text").str.replace("a", "_", n=0).alias("replaced_0"),
            pl.col("text").str.replace("a", "_", n=-1).alias("replaced_minus1"),
        ),
    )


def test_strip_chars() -> None:
    assert_eq(
        (pql.col("text").str.strip_chars().alias("stripped")),
        (nw.col("text").str.strip_chars().alias("stripped")),
    )
    assert_eq(
        (pql.col("text").str.strip_chars(" ").alias("stripped_space")),
        (nw.col("text").str.strip_chars(" ").alias("stripped_space")),
    )


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
        (pql.col("text").str.strip_chars_end().alias("rstripped")),
        (pl.col("text").str.strip_chars_end().alias("rstripped")),
    )
    assert_eq_pl(
        (pql.col("text").str.strip_chars_end(" ").alias("rstripped")),
        (pl.col("text").str.strip_chars_end(" ").alias("rstripped")),
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
    assert_eq_pl(
        (pql.col("text").str.len_bytes().alias("bytes")),
        (pl.col("text").str.len_bytes().alias("bytes")),
    )


def test_head_str() -> None:
    assert_eq(
        (pql.col("text").str.head(3).alias("first_3")),
        (nw.col("text").str.head(3).alias("first_3")),
    )


def test_tail_str() -> None:
    assert_eq(
        (pql.col("text").str.tail(3).alias("last_3")),
        (nw.col("text").str.tail(3).alias("last_3")),
    )


def test_reverse_str() -> None:
    assert_eq_pl(
        (pql.col("text").str.reverse().alias("reversed")),
        (pl.col("text").str.reverse().alias("reversed")),
    )


def test_to_titlecase() -> None:
    assert_eq(
        (pql.col("text").str.to_titlecase().alias("title")),
        (nw.col("text").str.to_titlecase().alias("title")),
    )


def test_split() -> None:
    assert_eq(
        (pql.col("text").str.split(",").alias("split")),
        (nw.col("text").str.split(",").alias("split")),
    )


def test_extract_all() -> None:
    assert_eq_pl(
        (pql.col("text").str.extract_all(r"\d+").alias("extracted")),
        (pl.col("text").str.extract_all(r"\d+").alias("extracted")),
    )


def test_count_matches() -> None:
    assert_eq_pl(
        (pql.col("text").str.count_matches("a", literal=True).alias("count")),
        (pl.col("text").str.count_matches("a", literal=True).alias("count")),
    )


def test_strip_prefix() -> None:
    assert_eq_pl(
        (pql.col("prefixed").str.strip_prefix("prefix_").alias("stripped")),
        (pl.col("prefixed").str.strip_prefix("prefix_").alias("stripped")),
    )
    assert_eq_pl(
        (
            pql.col("prefixed")
            .str.strip_prefix(pql.col("prefix_col"))
            .alias("stripped"),
        ),
        (pl.col("prefixed").str.strip_prefix(pl.col("prefix_col")).alias("stripped"),),
    )
    assert_eq_pl(
        (pql.col("prefix_exact").str.strip_prefix("foo").alias("stripped")),
        (pl.col("prefix_exact").str.strip_prefix("foo").alias("stripped")),
    )


def test_strip_suffix() -> None:
    assert_eq_pl(
        (pql.col("suffixed").str.strip_suffix("_suffix").alias("stripped")),
        (pl.col("suffixed").str.strip_suffix("_suffix").alias("stripped")),
    )
    assert_eq_pl(
        (
            pql.col("suffixed")
            .str.strip_suffix(pql.col("suffix_col"))
            .alias("stripped"),
        ),
        (pl.col("suffixed").str.strip_suffix(pl.col("suffix_col")).alias("stripped"),),
    )
    assert_eq_pl(
        (pql.col("suffix_exact").str.strip_suffix("bar").alias("stripped")),
        (pl.col("suffix_exact").str.strip_suffix("bar").alias("stripped")),
    )


def test_replace_all() -> None:
    assert_eq(
        (pql.col("text").str.replace_all("o", "0", literal=True).alias("replaced")),
        (nw.col("text").str.replace_all("o", "0", literal=True).alias("replaced")),
    )

    assert_eq(
        (pql.col("text").str.replace_all("l", "L", literal=True).alias("replaced")),
        (nw.col("text").str.replace_all("l", "L", literal=True).alias("replaced")),
    )

    assert_eq(
        (pql.col("text").str.replace_all(r"\d+", "X", literal=False).alias("replaced")),
        (nw.col("text").str.replace_all(r"\d+", "X", literal=False).alias("replaced")),
    )
    assert_eq(
        (
            pql.col("text")
            .str.replace_all("suffix", pql.col("suffix_val"), literal=True)
            .alias("replaced"),
        ),
        (
            nw.col("text")
            .str.replace_all("suffix", nw.col("suffix_val"), literal=True)
            .alias("replaced"),
        ),
    )


def test_head() -> None:
    assert_eq(
        (pql.col("text").str.head(2).alias("first")),
        (nw.col("text").str.head(2).alias("first")),
    )


def test_tail() -> None:
    assert_eq(
        pql.col("text").str.tail(2).alias("last"),
        (nw.col("text").str.tail(2).alias("last")),
    )


def test_contains_regex() -> None:
    assert_eq(
        (pql.col("text").str.contains(r"\d+", literal=False).alias("has_digit")),
        (nw.col("text").str.contains(r"\d+", literal=False).alias("has_digit")),
    )


def test_count_matches_literal() -> None:
    assert_eq_pl(
        (pql.col("text").str.count_matches("a", literal=True).alias("count")),
        (pl.col("text").str.count_matches("a", literal=True).alias("count")),
    )


def test_count_matches_regex() -> None:
    assert_eq_pl(
        (pql.col("text").str.count_matches(r"\d+", literal=False).alias("count")),
        (pl.col("text").str.count_matches(r"\d+", literal=False).alias("count")),
    )


def test_to_date() -> None:
    assert_eq(
        (pql.col("date_str").str.to_date("%Y-%m-%d").alias("date")),
        (nw.col("date_str").str.to_date("%Y-%m-%d").alias("date")),
    )
    assert_eq(
        (pql.col("date_str").str.to_date().alias("date")),
        (nw.col("date_str").str.to_date().alias("date")),
    )


def test_to_datetime() -> None:
    assert_eq(
        (pql.col("dt_str").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("dt")),
        (nw.col("dt_str").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("dt")),
    )
    assert_eq_pl(
        (pql.col("dt_str").str.to_datetime().alias("dt")),
        (pl.col("dt_str").str.to_datetime().alias("dt")),
    )


def test_to_time() -> None:
    assert_eq_pl(
        (pql.col("time_str").str.to_time("%H:%M:%S").alias("time")),
        (pl.col("time_str").str.to_time("%H:%M:%S").alias("time")),
    )
    assert_eq_pl(
        (pql.col("time_str").str.to_time().alias("time")),
        (pl.col("time_str").str.to_time().alias("time")),
    )


def test_to_decimal() -> None:
    assert_eq_pl(
        (pql.col("numbers").str.to_decimal(scale=10).alias("dec")),
        (pl.col("numbers").str.to_decimal(scale=10).alias("dec")),
    )


def test_pad_start() -> None:
    assert_eq_pl(
        (pql.col("text_short").str.pad_start(5).alias("padded")),
        (pl.col("text_short").str.pad_start(5).alias("padded")),
    )
    assert_eq_pl(
        (pql.col("text_short").str.pad_start(10, fill_char="*").alias("padded")),
        (pl.col("text_short").str.pad_start(10, fill_char="*").alias("padded")),
    )


def test_pad_end() -> None:
    assert_eq_pl(
        (pql.col("text_short").str.pad_end(5).alias("padded")),
        (pl.col("text_short").str.pad_end(5).alias("padded")),
    )
    assert_eq_pl(
        (pql.col("text_short").str.pad_end(10, fill_char="-").alias("padded")),
        (pl.col("text_short").str.pad_end(10, fill_char="-").alias("padded")),
    )


def test_zfill() -> None:
    assert_eq_pl(
        (pql.col("numbers").str.zfill(10).alias("zfilled")),
        (pl.col("numbers").str.zfill(10).alias("zfilled")),
    )
