from functools import partial

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pql

assert_eq = partial(assert_frame_equal, check_dtypes=False, check_row_order=False)


@pytest.fixture
def sample_df_strings() -> pl.DataFrame:
    """Create a sample DataFrame with string data for testing."""
    return pl.DataFrame(
        {
            "text": [
                "  Hello World suffix  ",
                "  foo bar baz suffix  ",
                "  Polars is great suffix  ",
                "  Testing string functions suffix  ",
            ]
        }
    )


def test_to_uppercase(sample_df_strings: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_strings)
        .select(
            pql.col("text"),
            pql.col("text").str.to_uppercase().alias("upper"),
        )
        .collect(),
        sample_df_strings.lazy()
        .select(
            pl.col("text"),
            pl.col("text").str.to_uppercase().alias("upper"),
        )
        .collect(),
    )


def test_to_lowercase(sample_df_strings: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_strings)
        .select(
            pql.col("text"),
            pql.col("text").str.to_lowercase().alias("lower"),
        )
        .collect(),
        sample_df_strings.lazy()
        .select(
            pl.col("text"),
            pl.col("text").str.to_lowercase().alias("lower"),
        )
        .collect(),
    )


def test_len_chars(sample_df_strings: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_strings)
        .select(
            pql.col("text"),
            pql.col("text").str.len_chars().alias("length"),
        )
        .collect(),
        sample_df_strings.lazy()
        .select(
            pl.col("text"),
            pl.col("text").str.len_chars().alias("length"),
        )
        .collect(),
    )


def test_contains_literal(sample_df_strings: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_strings)
        .select(
            pql.col("text"),
            pql.col("text").str.contains("lo", literal=True).alias("contains_lo"),
        )
        .collect(),
        sample_df_strings.lazy()
        .select(
            pl.col("text"),
            pl.col("text").str.contains("lo", literal=True).alias("contains_lo"),
        )
        .collect(),
    )


def test_starts_with(sample_df_strings: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_strings)
        .select(
            pql.col("text"),
            pql.col("text").str.starts_with("Hello").alias("starts_hello"),
        )
        .collect(),
        sample_df_strings.lazy()
        .select(
            pl.col("text"),
            pl.col("text").str.starts_with("Hello").alias("starts_hello"),
        )
        .collect(),
    )


def test_ends_with(sample_df_strings: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_strings)
        .select(
            pql.col("text"),
            pql.col("text").str.ends_with("suffix").alias("ends_suffix"),
        )
        .collect(),
        sample_df_strings.lazy()
        .select(
            pl.col("text"),
            pl.col("text").str.ends_with("suffix").alias("ends_suffix"),
        )
        .collect(),
    )


def test_replace(sample_df_strings: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_strings)
        .select(
            pql.col("text"),
            pql.col("text").str.replace("Hello", "Hi").alias("replaced"),
        )
        .collect(),
        sample_df_strings.lazy()
        .select(
            pl.col("text"),
            pl.col("text").str.replace("Hello", "Hi", literal=True).alias("replaced"),
        )
        .collect(),
    )


def test_strip(sample_df_strings: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_strings)
        .select(
            pql.col("text"),
            pql.col("text").str.strip_chars().alias("stripped"),
        )
        .collect(),
        sample_df_strings.lazy()
        .select(
            pl.col("text"),
            pl.col("text").str.strip_chars().alias("stripped"),
        )
        .collect(),
    )


def test_strip_chars_start(sample_df_strings: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_strings)
        .select(
            pql.col("text"),
            pql.col("text").str.strip_chars_start().alias("lstripped"),
        )
        .collect(),
        sample_df_strings.lazy()
        .select(
            pl.col("text"),
            pl.col("text").str.strip_chars_start().alias("lstripped"),
        )
        .collect(),
    )


def test_strip_chars_end(sample_df_strings: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_strings)
        .select(
            pql.col("text"),
            pql.col("text").str.strip_chars_end().alias("rstripped"),
        )
        .collect(),
        sample_df_strings.lazy()
        .select(
            pl.col("text"),
            pl.col("text").str.strip_chars_end().alias("rstripped"),
        )
        .collect(),
    )


def test_slice(sample_df_strings: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_strings)
        .select(
            pql.col("text"),
            pql.col("text").str.slice(0, 5).alias("sliced"),
        )
        .collect(),
        sample_df_strings.lazy()
        .select(
            pl.col("text"),
            pl.col("text").str.slice(0, 5).alias("sliced"),
        )
        .collect(),
    )


def test_len_bytes() -> None:
    str_df = pl.DataFrame({"text": ["hello", "world", "café"]})
    result = (
        pql.LazyFrame(str_df)
        .select(pql.col("text").str.len_bytes().alias("bytes"))
        .collect()
    )
    expected = (
        str_df.lazy().select(pl.col("text").str.len_bytes().alias("bytes")).collect()
    )
    assert_eq(result, expected)


def test_pad_start() -> None:
    str_df = pl.DataFrame({"text": ["a", "bb", "ccc"]})
    result = (
        pql.LazyFrame(str_df)
        .select(pql.col("text").str.pad_start(5, "*").alias("padded"))
        .collect()
    )
    expected = (
        str_df.lazy()
        .select(pl.col("text").str.pad_start(5, "*").alias("padded"))
        .collect()
    )
    assert_eq(result, expected)


def test_pad_end() -> None:
    str_df = pl.DataFrame({"text": ["a", "bb", "ccc"]})
    result = (
        pql.LazyFrame(str_df)
        .select(pql.col("text").str.pad_end(5, "*").alias("padded"))
        .collect()
    )
    expected = (
        str_df.lazy()
        .select(pl.col("text").str.pad_end(5, "*").alias("padded"))
        .collect()
    )
    assert_eq(result, expected)


def test_zfill() -> None:
    str_df = pl.DataFrame({"text": ["1", "22", "333"]})
    result = (
        pql.LazyFrame(str_df)
        .select(pql.col("text").str.zfill(5).alias("zfilled"))
        .collect()
    )
    expected = (
        str_df.lazy().select(pl.col("text").str.zfill(5).alias("zfilled")).collect()
    )
    assert_eq(result, expected)


def test_head_str() -> None:
    str_df = pl.DataFrame({"text": ["hello", "world"]})
    result = (
        pql.LazyFrame(str_df)
        .select(pql.col("text").str.head(3).alias("first_3"))
        .collect()
    )
    expected = (
        str_df.lazy().select(pl.col("text").str.head(3).alias("first_3")).collect()
    )
    assert_eq(result, expected)


def test_tail_str() -> None:
    str_df = pl.DataFrame({"text": ["hello", "world"]})
    result = (
        pql.LazyFrame(str_df)
        .select(pql.col("text").str.tail(3).alias("last_3"))
        .collect()
    )
    expected = (
        str_df.lazy().select(pl.col("text").str.tail(3).alias("last_3")).collect()
    )
    assert_eq(result, expected)


def test_reverse_str() -> None:
    str_df = pl.DataFrame({"text": ["hello", "world"]})
    result = (
        pql.LazyFrame(str_df)
        .select(pql.col("text").str.reverse().alias("reversed"))
        .collect()
    )
    expected = (
        str_df.lazy().select(pl.col("text").str.reverse().alias("reversed")).collect()
    )
    assert_eq(result, expected)


def test_to_titlecase() -> None:
    str_df = pl.DataFrame({"text": ["hello world", "foo bar"]})
    result = (
        pql.LazyFrame(str_df)
        .select(pql.col("text").str.to_titlecase().alias("title"))
        .collect()
    )
    expected = (
        str_df.lazy().select(pl.col("text").str.to_titlecase().alias("title")).collect()
    )
    assert_eq(result, expected)


def test_split() -> None:
    df = pl.DataFrame({"text": ["a,b,c", "x,y"]})
    result = (
        pql.LazyFrame(df)
        .select(pql.col("text").str.split(",").alias("split"))
        .collect()
    )
    expected = df.lazy().select(pl.col("text").str.split(",").alias("split")).collect()
    assert_eq(result, expected)


def test_extract() -> None:
    df = pl.DataFrame({"text": ["abc123", "xyz456"]})
    result = (
        pql.LazyFrame(df)
        .select(pql.col("text").str.extract(r"(\d+)").alias("extracted"))
        .collect()
    )
    expected = (
        df.lazy()
        .select(pl.col("text").str.extract(r"(\d+)").alias("extracted"))
        .collect()
    )
    assert_eq(result, expected)


def test_extract_all() -> None:
    df = pl.DataFrame({"text": ["abc123def456"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.extract_all(r"\d+").alias("extracted"))
        .collect(),
        df.lazy()
        .select(pl.col("text").str.extract_all(r"\d+").alias("extracted"))
        .collect(),
    )


def test_count_matches() -> None:
    df = pl.DataFrame({"text": ["abcabc", "xyzxyz"]})

    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.count_matches("a", literal=True).alias("count"))
        .collect(),
        df.lazy()
        .select(pl.col("text").str.count_matches("a", literal=True).alias("count"))
        .collect(),
    )


def test_strip_prefix() -> None:
    df = pl.DataFrame({"text": ["prefix_text", "prefix_other"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.strip_prefix("prefix_").alias("stripped"))
        .collect(),
        df.lazy()
        .select(pl.col("text").str.strip_prefix("prefix_").alias("stripped"))
        .collect(),
    )


def test_strip_suffix() -> None:
    df = pl.DataFrame({"text": ["text_suffix", "other_suffix"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.strip_suffix("_suffix").alias("stripped"))
        .collect(),
        df.lazy()
        .select(pl.col("text").str.strip_suffix("_suffix").alias("stripped"))
        .collect(),
    )


def test_replace_all() -> None:
    df = pl.DataFrame({"text": ["hello world", "foo bar"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(
            pql.col("text").str.replace_all("o", "0", literal=True).alias("replaced")
        )
        .collect(),
        df.lazy()
        .select(
            pl.col("text").str.replace_all("o", "0", literal=True).alias("replaced")
        )
        .collect(),
    )


def test_find() -> None:
    df = pl.DataFrame({"text": ["hello", "world"]})

    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.find("l", literal=True).alias("index"))
        .collect(),
        df.lazy()
        .select(pl.col("text").str.find("l", literal=True).alias("index"))
        .collect(),
    )


def test_head() -> None:
    """Test string head."""
    df = pl.DataFrame({"text": ["Hello", "World"]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("text").str.head(2).alias("first")).collect(),
        pl.LazyFrame(df).select(pl.col("text").str.head(2).alias("first")).collect(),
    )


def test_tail() -> None:
    """Test string tail."""
    df = pl.DataFrame({"text": ["Hello", "World"]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("text").str.tail(2).alias("last")).collect(),
        pl.LazyFrame(df).select(pl.col("text").str.tail(2).alias("last")).collect(),
    )


def test_contains_regex() -> None:
    """Test string contains with regex."""
    df = pl.DataFrame({"text": ["hello123", "world", "test456"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.contains(r"\d+", literal=False).alias("has_digit"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("text").str.contains(r"\d+", literal=False).alias("has_digit"))
        .collect(),
    )


def test_strip_chars_none() -> None:
    """Test string strip_chars without characters."""
    df = pl.DataFrame({"text": ["  hello  ", "  world  "]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.strip_chars().alias("stripped"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("text").str.strip_chars().alias("stripped"))
        .collect(),
    )


def test_count_matches_literal() -> None:
    """Test string count_matches with literal."""
    df = pl.DataFrame({"text": ["aaa", "ababab"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.count_matches("a", literal=True).alias("count"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("text").str.count_matches("a", literal=True).alias("count"))
        .collect(),
    )


def test_count_matches_regex() -> None:
    """Test string count_matches with regex."""
    df = pl.DataFrame({"text": ["hello123world456", "test"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.count_matches(r"\d+", literal=False).alias("count"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("text").str.count_matches(r"\d+", literal=False).alias("count"))
        .collect(),
    )


def test_to_date() -> None:
    """Test string to_date."""
    df = pl.DataFrame({"date_str": ["2024-01-15", "2024-02-20"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("date_str").str.to_date("%Y-%m-%d").alias("date"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("date_str").str.to_date("%Y-%m-%d").alias("date"))
        .collect(),
    )


def test_to_datetime() -> None:
    """Test string to_datetime."""
    df = pl.DataFrame({"dt_str": ["2024-01-15 10:30:00"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("dt_str").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("dt"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("dt_str").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("dt"))
        .collect(),
    )


def test_to_time() -> None:
    """Test string to_time."""
    df = pl.DataFrame({"time_str": ["10:30:00", "15:45:30"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("time_str").str.to_time("%H:%M:%S").alias("time"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("time_str").str.to_time("%H:%M:%S").alias("time"))
        .collect(),
    )


def test_json_path_match() -> None:
    """Test str.json_path_match."""
    df = pl.DataFrame({"json": ['{"a": 1}', '{"a": 2}']})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("json").str.json_path_match("$.a").alias("val"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("json").str.json_path_match("$.a").alias("val"))
        .collect(),
    )


def test_json_decode() -> None:
    """Test json_decode."""
    df = pl.DataFrame({"json_str": "1"})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("json_str").str.json_decode(pql.Int64).alias("decoded"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("json_str").str.json_decode(pl.Int64).alias("decoded"))
        .collect(),
    )


def test_to_decimal() -> None:
    """Test str.to_decimal."""
    df = pl.DataFrame({"text": ["123.45", "67.89"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.to_decimal(scale=10).alias("dec"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("text").str.to_decimal(scale=10).alias("dec"))
        .collect(),
    )


def test_replace_all_literal() -> None:
    """Test str.replace_all with literal."""
    df = pl.DataFrame({"text": ["hello hello", "world world"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(
            pql.col("text").str.replace_all("l", "L", literal=True).alias("replaced")
        )
        .collect(),
        pl.LazyFrame(df)
        .select(
            pl.col("text").str.replace_all("l", "L", literal=True).alias("replaced")
        )
        .collect(),
    )


def test_replace_all_regex() -> None:
    """Test str.replace_all with regex."""
    df = pl.DataFrame({"text": ["hello123", "world456"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(
            pql.col("text")
            .str.replace_all(r"\d+", "X", literal=False)
            .alias("replaced")
        )
        .collect(),
        pl.LazyFrame(df)
        .select(
            pl.col("text").str.replace_all(r"\d+", "X", literal=False).alias("replaced")
        )
        .collect(),
    )


def test_find_literal() -> None:
    """Test str.find with literal=True."""
    df = pl.DataFrame({"text": ["hello world", "foo bar baz"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.find("o", literal=True).alias("pos"))
        .collect(),
        pl.LazyFrame(df)
        .select((pl.col("text").str.find("o", literal=True)).alias("pos"))
        .collect(),
    )


def test_find_regex() -> None:
    """Test str.find with literal=False (regex)."""
    df = pl.DataFrame({"text": ["abc123def", "xyz456"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.find(r"\d", literal=False).alias("pos"))
        .collect(),
        pl.LazyFrame(df)
        .select((pl.col("text").str.find(r"\d", literal=False)).alias("pos"))
        .collect(),
    )


def test_replace_many() -> None:
    """Test str.replace_many."""
    df = pl.DataFrame({"text": ["hello world", "foo bar"]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("text").str.replace_many(["h"], "H").alias("replaced"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("text").str.replace_many(["h"], "H").alias("replaced"))
        .collect(),
    )
