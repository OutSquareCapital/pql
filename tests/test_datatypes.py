from __future__ import annotations

import enum

import duckdb
from duckdb import sqltypes
from polars.testing import assert_frame_equal

import pql


def test_expr_cast_numeric_and_string_schema() -> None:
    source = pql.LazyFrame(
        {
            "x": [1, 2, 3],
            "y": [[1, 2], [3, 4], [5, 6]],
            "z": [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
            "dates": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "hours": [
                "2021-01-01 00:00:00",
                "2021-01-01 01:00:00",
                "2021-01-01 02:00:00",
            ],
            "blobs": [b"foo", b"bar", b"baz"],
            "nanoseconds": [
                "2021-01-01 00:00:00.000000000",
                "2021-01-01 01:00:00.000000000",
                "2021-01-01 02:00:00.000000000",
            ],
            "time": ["12:00:00", "13:00:00", "14:00:00"],
            "duration": ["1 day", "2 days", "3 days"],
        }
    )

    casted = source.select(
        pql.col("x").cast(pql.Int8()).alias("i8"),
        pql.col("x").cast(pql.Int16()).alias("i16"),
        pql.col("x").cast(pql.Int32()).alias("i32"),
        pql.col("x").cast(pql.Int64()).alias("i64"),
        pql.col("x").cast(pql.Int128()).alias("i128"),
        pql.col("x").cast(pql.UInt8()).alias("u8"),
        pql.col("x").cast(pql.UInt16()).alias("u16"),
        pql.col("x").cast(pql.UInt32()).alias("u32"),
        pql.col("x").cast(pql.UInt64()).alias("u64"),
        pql.col("x").cast(pql.UInt128()).alias("u128"),
        pql.col("x").cast(pql.Float32()).alias("f32"),
        pql.col("x").cast(pql.Float64()).alias("f64"),
        pql.col("x").cast(pql.Boolean()).alias("bool"),
        pql.col("x").cast(pql.Decimal(10, 2)).alias("dec"),
        pql.col("x").cast(pql.String()).alias("s"),
        pql.col("time").cast(pql.Time()).alias("time"),
        pql.col("dates").cast(pql.Date()).alias("dates"),
        pql.col("hours").cast(pql.Datetime(time_zone="UTC")).alias("hours"),
        pql.col("nanoseconds").cast(pql.Datetime(time_unit="ns")).alias("nanoseconds"),
        pql.col("y").cast(pql.List(pql.UInt16())).alias("lst"),
        pql.col("y").cast(pql.Array(pql.UInt16(), shape=(2, 3))).alias("arr"),
        pql.col("blobs").cast(pql.Binary()).alias("blobs"),
        pql.col("duration").cast(pql.Duration()).alias("duration"),
    )
    schema = casted.schema
    assert isinstance(schema["i8"], pql.Int8)
    assert isinstance(schema["i16"], pql.Int16)
    assert isinstance(schema["i32"], pql.Int32)
    assert isinstance(schema["i64"], pql.Int64)
    assert isinstance(schema["u8"], pql.UInt8)
    assert isinstance(schema["u16"], pql.UInt16)
    assert isinstance(schema["u32"], pql.UInt32)
    assert isinstance(schema["u64"], pql.UInt64)
    assert isinstance(schema["f32"], pql.Float32)
    assert isinstance(schema["f64"], pql.Float64)
    assert isinstance(schema["bool"], pql.Boolean)
    assert isinstance(schema["dec"], pql.Decimal)
    assert isinstance(schema["time"], pql.Time)
    assert isinstance(schema["s"], pql.String)
    assert isinstance(schema["arr"], pql.Array)
    assert isinstance(schema["arr"].inner, pql.UInt16)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert tuple(schema["arr"].shape) == (2, 3)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
    assert isinstance(schema["lst"], pql.List)
    assert isinstance(schema["lst"].inner, pql.UInt16)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert isinstance(schema["dates"], pql.Date)
    assert isinstance(schema["hours"], pql.Datetime)
    assert schema["nanoseconds"].time_unit == "ns"  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    assert isinstance(schema["blobs"], pql.Binary)
    assert isinstance(schema["duration"], pql.Duration)


def test_schema_nested_types_from_casts() -> None:
    source = pql.LazyFrame(
        {
            "json_obj": ['{"a": 1, "b": [1,2]}', None],
            "arr_text": ["[1,2,3]", None],
        }
    )

    casted = source.select(
        pql.col("json_obj")
        .cast(
            pql.Struct(
                {
                    "a": pql.Int32(),
                    "b": pql.List(pql.Int32()),
                }
            )
        )
        .alias("obj"),
        pql.col("arr_text").cast(pql.List(pql.Int32())).alias("arr"),
    )

    obj_dtype = casted.schema["obj"]
    arr_dtype = casted.schema["arr"]

    assert isinstance(obj_dtype, pql.Struct)
    assert isinstance(obj_dtype.fields["a"], pql.Int32)
    assert isinstance(obj_dtype.fields["b"], pql.List)
    assert isinstance(obj_dtype.fields["b"].inner, pql.Int32)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    assert isinstance(arr_dtype, pql.List)
    assert isinstance(arr_dtype.inner, pql.Int32)


def test_enum_dtype_behaves_like_string_cast() -> None:
    source = pql.LazyFrame({"label": ["A", "B", None]})

    enum_cast = source.select(pql.col("label").cast(pql.Enum(["A", "B"]))).collect()
    string_cast = source.select(pql.col("label").cast(pql.String())).collect()

    assert_frame_equal(enum_cast, string_cast)


def test_array_and_enum_sql_paths() -> None:
    class _Status(enum.Enum):
        A = "A"
        B = "B"

    enum_dtype = pql.Enum(_Status)
    array_dtype = pql.Array(pql.Int32(), 4)

    assert enum_dtype.sql() == sqltypes.VARCHAR
    assert "A" in enum_dtype.categories
    assert "B" in enum_dtype.categories
    assert array_dtype.sql().id == duckdb.array_type(sqltypes.INTEGER, 4).id
