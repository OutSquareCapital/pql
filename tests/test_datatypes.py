from __future__ import annotations

from enum import Enum

import pql


def test_expr_cast_numeric_and_string_schema() -> None:
    class MyEnum(Enum):
        A = "A"
        B = "B"
        C = "C"

    source = pql.LazyFrame(
        {
            "x": [1, 2, 3],
            "1d": [[1, 2], [3, 4], [5, 6]],
            "2d": [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
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
            "enumerated": ["A", "B", "C"],
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
        pql.col("hours").cast(pql.DatetimeTZ()).alias("hours"),
        pql.col("nanoseconds").cast(pql.Datetime(time_unit="ns")).alias("nanoseconds"),
        pql.col("1d").cast(pql.List(pql.UInt16())).alias("lst"),
        pql.col("1d").cast(pql.Array(pql.UInt16(), shape=2)).alias("arr_1d"),
        pql.col("2d")
        .cast(pql.Array(pql.Array(pql.UInt16(), shape=2), shape=2))
        .alias("arr_2d"),
        pql.col("blobs").cast(pql.Binary()).alias("blobs"),
        pql.col("duration").cast(pql.Duration()).alias("duration"),
        pql.col("enumerated").cast(pql.Enum(["A", "B", "C"])).alias("enumerated"),
        pql.col("enumerated").cast(pql.Enum(MyEnum)).alias("enumerated_enum"),
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
    assert isinstance(schema["arr_1d"], pql.Array)
    assert isinstance(schema["arr_1d"].inner, pql.UInt16)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert schema["arr_1d"].shape == 2  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
    assert isinstance(schema["arr_2d"], pql.Array)
    assert schema["arr_2d"].shape == 2  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
    assert isinstance(schema["lst"], pql.List)
    assert isinstance(schema["lst"].inner, pql.UInt16)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert isinstance(schema["dates"], pql.Date)
    assert isinstance(schema["hours"], pql.DatetimeTZ)
    assert schema["nanoseconds"].time_unit == "ns"  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    assert isinstance(schema["blobs"], pql.Binary)
    assert isinstance(schema["duration"], pql.Duration)
    assert isinstance(schema["enumerated"], pql.Enum)
    assert tuple(schema["enumerated"].categories) == ("A", "B", "C")  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
    assert isinstance(schema["enumerated_enum"], pql.Enum)
    assert tuple(schema["enumerated_enum"].categories) == ("A", "B", "C")  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]


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
