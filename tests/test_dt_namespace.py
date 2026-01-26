from datetime import UTC, datetime
from functools import partial

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pql

assert_eq = partial(assert_frame_equal, check_dtypes=False, check_row_order=False)


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 28, 22],
            "salary": [50000.0, 60000.0, 75000.0, 55000.0, 45000.0],
            "department": [
                "Engineering",
                "Sales",
                "Engineering",
                "Sales",
                "Engineering",
            ],
            "is_active": [True, True, False, True, True],
        }
    )


def test_dt_minute() -> None:
    """Test datetime minute."""
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 3), "1h", eager=True
            )
        }
    )
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.minute().alias("minute")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.minute().alias("minute")).collect(),
    )


def test_dt_second() -> None:
    """Test datetime second."""
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 3), "1h", eager=True
            )
        }
    )
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.second().alias("second")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.second().alias("second")).collect(),
    )


def test_dt_microsecond() -> None:
    """Test datetime microsecond."""
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 3), "1h", eager=True
            )
        }
    )
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.microsecond().alias("microsecond"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.microsecond().alias("microsecond"))
        .collect(),
    )


def test_dt_nanosecond() -> None:
    """Test datetime nanosecond."""
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 3), "1h", eager=True
            )
        }
    )
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.nanosecond().alias("nanosecond"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.nanosecond().alias("nanosecond"))
        .collect(),
    )


def test_dt_weekday() -> None:
    """Test datetime weekday."""
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 7), "1d", eager=True
            )
        }
    )
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.weekday().alias("weekday")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.weekday().alias("weekday")).collect(),
    )


def test_dt_week() -> None:
    """Test datetime week."""
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 12, 31), "1mo", eager=True
            )
        }
    )
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.week().alias("week")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.week().alias("week")).collect(),
    )


def test_dt_truncate() -> None:
    """Test datetime truncate."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 15, 10, 30, 45)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.truncate("1d").alias("truncated"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.truncate("1d").alias("truncated"))
        .collect(),
    )


def test_dt_strftime() -> None:
    """Test datetime strftime."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 15, 10, 30)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.strftime("%Y-%m-%d").alias("formatted"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.strftime("%Y-%m-%d").alias("formatted"))
        .collect(),
    )


def test_dt_date() -> None:
    """Test datetime date extraction."""
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 3), "1d", eager=True
            )
        }
    )
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.date().alias("date")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.date().alias("date")).collect(),
    )


def test_dt_epoch_seconds() -> None:
    """Test epoch seconds."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.epoch("s").alias("epoch")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.epoch("s").alias("epoch")).collect(),
    )


def test_dt_epoch_ms() -> None:
    """Test epoch milliseconds."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.epoch("ms").alias("epoch")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.epoch("ms").alias("epoch")).collect(),
    )


def test_dt_epoch_us() -> None:
    """Test epoch microseconds."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.epoch("us").alias("epoch")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.epoch("us").alias("epoch")).collect(),
    )


def test_dt_epoch_ns() -> None:
    """Test epoch nanoseconds."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.epoch("ns").alias("epoch")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.epoch("ns").alias("epoch")).collect(),
    )


def test_dt_epoch_default() -> None:
    """Test epoch default (microseconds)."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.epoch().alias("epoch")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.epoch("us").alias("epoch")).collect(),
    )


def test_dt_year() -> None:
    """Test datetime year extraction."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 15), pl.datetime(2023, 12, 31)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.year().alias("year")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.year().alias("year")).collect(),
    )


def test_dt_month() -> None:
    """Test datetime month extraction."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 15), pl.datetime(2024, 12, 31)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.month().alias("month")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.month().alias("month")).collect(),
    )


def test_dt_day() -> None:
    """Test datetime day extraction."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 15), pl.datetime(2024, 1, 31)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.day().alias("day")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.day().alias("day")).collect(),
    )


def test_dt_hour() -> None:
    """Test datetime hour extraction."""
    df = pl.DataFrame(
        {"ts": [pl.datetime(2024, 1, 1, 10, 30), pl.datetime(2024, 1, 1, 23, 45)]}
    )
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.hour().alias("hour")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.hour().alias("hour")).collect(),
    )


def test_dt_quarter() -> None:
    """Test datetime quarter extraction."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1), pl.datetime(2024, 7, 1)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.quarter().alias("quarter")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.quarter().alias("quarter")).collect(),
    )


def test_dt_ordinal_day() -> None:
    """Test datetime ordinal day."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1), pl.datetime(2024, 12, 31)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.ordinal_day().alias("day_of_year"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.ordinal_day().alias("day_of_year"))
        .collect(),
    )


def test_dt_iso_year() -> None:
    """Test datetime ISO year."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1), pl.datetime(2023, 12, 31)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.iso_year().alias("iso_year"))
        .collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.iso_year().alias("iso_year")).collect(),
    )


def test_dt_total_microseconds() -> None:
    """Test dt.total_microseconds."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.total_microseconds().alias("us"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.total_microseconds().alias("us"))
        .collect(),
    )


def test_dt_total_nanoseconds() -> None:
    """Test dt.total_nanoseconds."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.total_nanoseconds().alias("ns"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.total_nanoseconds().alias("ns"))
        .collect(),
    )


def test_dt_timestamp() -> None:
    """Test dt.timestamp."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.timestamp().alias("ts_val"))
        .collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.timestamp().alias("ts_val")).collect(),
    )


def test_dt_offset_by() -> None:
    """Test dt.offset_by."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.offset_by("1d").alias("offset"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.offset_by("1d").alias("offset"))
        .collect(),
    )


def test_dt_century() -> None:
    """Test dt.century."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.century().alias("century")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.century().alias("century")).collect(),
    )


def test_dt_month_start() -> None:
    """Test dt.month_start."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 15)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.month_start().alias("start"))
        .collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.month_start().alias("start")).collect(),
    )


def test_dt_month_end() -> None:
    """Test dt.month_end."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 15)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.month_end().alias("end")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.month_end().alias("end")).collect(),
    )


def test_dt_is_leap_year_true() -> None:
    """Test dt.is_leap_year for leap years."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1), pl.datetime(2020, 6, 15)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.is_leap_year().alias("is_leap"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.is_leap_year().alias("is_leap"))
        .collect(),
    )


def test_convert_time_zone() -> None:
    """Test dt.convert_time_zone."""
    df = pl.DataFrame({"ts": [datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.convert_time_zone("America/New_York"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.convert_time_zone("America/New_York"))
        .collect(),
    )


def test_dt_is_leap_year_false() -> None:
    """Test dt.is_leap_year for non-leap years."""
    df = pl.DataFrame({"ts": [pl.datetime(2023, 1, 1), pl.datetime(2021, 6, 15)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.is_leap_year().alias("is_leap"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.is_leap_year().alias("is_leap"))
        .collect(),
    )


def test_dt_millisecond_extraction() -> None:
    """Test dt.millisecond extraction."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1, 10, 30, 45, 123456)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.millisecond().alias("ms")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.millisecond().alias("ms")).collect(),
    )


def test_dt_dst_offset() -> None:
    """Test dt.dst_offset."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df).select(pql.col("ts").dt.dst_offset().alias("dst")).collect(),
        pl.LazyFrame(df).select(pl.col("ts").dt.dst_offset().alias("dst")).collect(),
    )


def test_dt_base_utc_offset() -> None:
    """Test dt.base_utc_offset."""
    df = pl.DataFrame({"ts": [pl.datetime(2024, 1, 1)]})
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("ts").dt.base_utc_offset().alias("offset"))
        .collect(),
        pl.LazyFrame(df)
        .select(pl.col("ts").dt.base_utc_offset().alias("offset"))
        .collect(),
    )


def test_year(sample_df_dates: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_dates)
        .select(
            pql.col("ts").dt.year().alias("year"),
        )
        .collect(),
        sample_df_dates.lazy()
        .select(
            pl.col("ts").dt.year().alias("year"),
        )
        .collect(),
    )


def test_month(sample_df_dates: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_dates)
        .select(
            pql.col("ts").dt.month().alias("month"),
        )
        .collect(),
        sample_df_dates.lazy()
        .select(
            pl.col("ts").dt.month().alias("month"),
        )
        .collect(),
    )


def test_day(sample_df_dates: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_dates)
        .select(
            pql.col("ts").dt.day().alias("day"),
        )
        .collect(),
        sample_df_dates.lazy()
        .select(
            pl.col("ts").dt.day().alias("day"),
        )
        .collect(),
    )


def test_hour(sample_df_dates: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_dates)
        .select(
            pql.col("ts").dt.hour().alias("hour"),
        )
        .collect(),
        sample_df_dates.lazy()
        .select(
            pl.col("ts").dt.hour().alias("hour"),
        )
        .collect(),
    )


def test_quarter(sample_df_dates: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df_dates)
        .select(pql.col("ts").dt.quarter().alias("quarter"))
        .collect()
    )
    expected = (
        sample_df_dates.lazy()
        .select(pl.col("ts").dt.quarter().alias("quarter"))
        .collect()
    )
    assert_eq(result, expected)


def test_ordinal_day(sample_df_dates: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df_dates)
        .select(pql.col("ts").dt.ordinal_day().alias("day_of_year"))
        .collect()
    )
    expected = (
        sample_df_dates.lazy()
        .select(pl.col("ts").dt.ordinal_day().alias("day_of_year"))
        .collect()
    )
    assert_eq(result, expected)


def test_iso_year(sample_df_dates: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df_dates)
        .select(pql.col("ts").dt.iso_year().alias("iso_year"))
        .collect()
    )
    expected = (
        sample_df_dates.lazy()
        .select(pl.col("ts").dt.iso_year().alias("iso_year"))
        .collect()
    )
    assert_eq(result, expected)


def test_millisecond(sample_df_dates: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df_dates)
        .select(pql.col("ts").dt.millisecond().alias("ms"))
        .collect()
    )
    expected = (
        sample_df_dates.lazy()
        .select(pl.col("ts").dt.millisecond().alias("ms"))
        .collect()
    )
    assert_eq(result, expected)


def test_microsecond(sample_df_dates: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df_dates)
        .select(pql.col("ts").dt.microsecond().alias("us"))
        .collect()
    )
    expected = (
        sample_df_dates.lazy()
        .select(pl.col("ts").dt.microsecond().alias("us"))
        .collect()
    )
    assert_eq(result, expected)


def test_epoch(sample_df_dates: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df_dates)
        .select(pql.col("ts").dt.epoch("s").alias("epoch_s"))
        .collect()
    )
    expected = (
        sample_df_dates.lazy()
        .select(pl.col("ts").dt.epoch("s").alias("epoch_s"))
        .collect()
    )
    assert_eq(result, expected)


def test_truncate(sample_df_dates: pl.DataFrame) -> None:
    result = (
        pql.LazyFrame(sample_df_dates)
        .select(pql.col("ts").dt.truncate("day").alias("truncated"))
        .collect()
    )
    expected = (
        sample_df_dates.lazy()
        .select(pl.col("ts").dt.truncate("1d").alias("truncated"))
        .collect()
    )
    assert_eq(result, expected)


def test_minute(sample_df_dates: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_dates)
        .select(pql.col("ts").dt.minute().alias("minute"))
        .collect(),
        sample_df_dates.lazy()
        .select(pl.col("ts").dt.minute().alias("minute"))
        .collect(),
    )


def test_second(sample_df_dates: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_dates)
        .select(pql.col("ts").dt.second().alias("second"))
        .collect(),
        sample_df_dates.lazy()
        .select(pl.col("ts").dt.second().alias("second"))
        .collect(),
    )


def test_weekday(sample_df_dates: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_dates)
        .select(pql.col("ts").dt.weekday().alias("weekday"))
        .collect(),
        sample_df_dates.lazy()
        .select(pl.col("ts").dt.weekday().alias("weekday"))
        .collect(),
    )


def test_week(sample_df_dates: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_dates)
        .select(
            pql.col("ts").dt.week().alias("week"),
        )
        .collect(),
        sample_df_dates.lazy()
        .select(
            pl.col("ts").dt.week().alias("week"),
        )
        .collect(),
    )


def test_date(sample_df_dates: pl.DataFrame) -> None:
    assert_eq(
        pql.LazyFrame(sample_df_dates)
        .select(pql.col("ts").dt.date().alias("date_only"))
        .collect(),
        sample_df_dates.lazy()
        .select(pl.col("ts").dt.date().alias("date_only"))
        .collect(),
    )


def test_timestamp() -> None:
    df = pl.DataFrame(
        {"dt": [datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 15, tzinfo=UTC)]}
    )
    assert_eq(
        pql.LazyFrame(df)
        .select(pql.col("dt").dt.timestamp().alias("timestamp"))
        .collect(),
        df.lazy().select(pl.col("dt").dt.timestamp().alias("timestamp")).collect(),
    )
