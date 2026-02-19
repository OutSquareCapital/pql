"""Data types Mapping for PQL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum as PyEnum
from functools import partial
from typing import Literal, Self

import duckdb
import pyochain as pc
from duckdb import sqltypes
from duckdb.sqltypes import DuckDBPyType

from .sql import (
    ArrayType,
    DecimalType,
    EnumType,
    ListType,
    RawTypes,
    SqlType,
    StructType,
)

TimeUnit = Literal["s", "ms", "us", "ns"]
PRECISION_MAP: dict[TimeUnit, DuckDBPyType] = {
    "s": sqltypes.TIMESTAMP_S,
    "ms": sqltypes.TIMESTAMP_MS,
    "us": sqltypes.TIMESTAMP,
    "ns": sqltypes.TIMESTAMP_NS,
}


@dataclass(slots=True)
class DataType(ABC):
    """Base class for data types."""

    @abstractmethod
    def sql(self) -> DuckDBPyType:
        """Inner data type representation."""
        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def from_duckdb(dtype: SqlType) -> DataType:
        match dtype:
            case ListType():
                return List(DataType.from_duckdb(dtype.child.dtype))
            case StructType():
                return (
                    pc.Iter(dtype.fields)
                    .map(
                        lambda field: (
                            field.name,
                            DataType.from_duckdb(field.dtype),
                        ),
                    )
                    .into(Struct)
                )
            case ArrayType():
                return Array(DataType.from_duckdb(dtype.child.dtype), dtype.size.value)

            case EnumType():
                return Enum(dtype.child.values)
            case DecimalType():
                return Decimal(dtype.precision.value, dtype.scale.value)
            case _:
                return (
                    NON_NESTED_MAP.get_item(dtype.type_id)
                    .map(lambda dt: dt())
                    .expect(f"Unsupported data type: {dtype}")
                )


@dataclass(slots=True)
class Binary(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.BLOB


@dataclass(slots=True)
class Time(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.TIME


@dataclass(slots=True)
class Duration(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.INTERVAL


@dataclass(slots=True)
class Boolean(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.BOOLEAN


@dataclass(slots=True)
class String(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.VARCHAR


@dataclass(slots=True)
class Date(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.DATE


@dataclass(slots=True)
class Float32(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.FLOAT


@dataclass(slots=True)
class Float64(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.DOUBLE


@dataclass(slots=True)
class Int8(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.TINYINT


@dataclass(slots=True)
class Int16(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.SMALLINT


@dataclass(slots=True)
class Int32(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.INTEGER


@dataclass(slots=True)
class Int64(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.BIGINT


@dataclass(slots=True)
class Int128(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.HUGEINT


@dataclass(slots=True)
class UInt8(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.UTINYINT


@dataclass(slots=True)
class UInt16(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.USMALLINT


@dataclass(slots=True)
class UInt32(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.UINTEGER


@dataclass(slots=True)
class UInt64(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.UBIGINT


@dataclass(slots=True)
class UInt128(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.UHUGEINT


@dataclass(slots=True)
class Datetime(DataType):
    time_unit: TimeUnit = "ns"

    def sql(self) -> DuckDBPyType:
        return PRECISION_MAP.get(self.time_unit, sqltypes.TIMESTAMP)


@dataclass(slots=True)
class DatetimeTZ(DataType):
    def sql(self) -> DuckDBPyType:
        return sqltypes.TIMESTAMP_TZ


@dataclass(slots=True)
class Decimal(DataType):
    precision: int = 18
    scale: int = 0

    def sql(self) -> DuckDBPyType:
        return duckdb.decimal_type(self.precision, self.scale)


@dataclass(slots=True)
class Array(DataType):
    inner: DataType
    shape: int

    def with_dim(self, shape: int) -> Self:
        """Add another level of nesting to the array."""
        return self.__class__(self.inner, shape)

    def sql(self) -> DuckDBPyType:
        return duckdb.array_type(self.inner.sql(), self.shape)


@dataclass(slots=True)
class Struct(DataType):
    fields: pc.Dict[str, DataType]

    def __init__(
        self, fields: Mapping[str, DataType] | Iterable[tuple[str, DataType]]
    ) -> None:
        self.fields = pc.Dict(fields)

    def sql(self) -> DuckDBPyType:
        return duckdb.struct_type(
            self.fields.items()
            .iter()
            .map_star(lambda name, col: (name, col.sql()))
            .collect(dict)
        )


@dataclass(slots=True)
class List(DataType):
    inner: DataType

    def sql(self) -> DuckDBPyType:
        return duckdb.list_type(self.inner.sql())


@dataclass(slots=True)
class Enum(DataType):
    categories: pc.Seq[str]

    def __init__(self, categories: Iterable[str] | type[PyEnum]) -> None:
        match categories:
            case type():
                cats = pc.Iter(categories).map(lambda i: i.value)
            case Iterable():
                cats = pc.Iter(categories)
        self.categories = cats.collect()

    def sql(self) -> DuckDBPyType:
        return DuckDBPyType(f"ENUM{self.categories.into(tuple)!r}")


NON_NESTED_MAP: pc.Dict[str, Callable[[], DataType]] = pc.Dict.from_ref(
    {
        RawTypes.HUGEINT: Int128,
        RawTypes.BIGINT: Int64,
        RawTypes.INTEGER: Int32,
        RawTypes.SMALLINT: Int16,
        RawTypes.TINYINT: Int8,
        RawTypes.UHUGEINT: UInt128,
        RawTypes.UBIGINT: UInt64,
        RawTypes.UINTEGER: UInt32,
        RawTypes.USMALLINT: UInt16,
        RawTypes.UTINYINT: UInt8,
        RawTypes.DOUBLE: Float64,
        RawTypes.FLOAT: Float32,
        RawTypes.VARCHAR: String,
        RawTypes.DATE: Date,
        RawTypes.TIMESTAMP_S: partial(Datetime, "s"),
        RawTypes.TIMESTAMP_MS: partial(Datetime, "ms"),
        RawTypes.TIMESTAMP: partial(Datetime),
        RawTypes.TIMESTAMP_NS: partial(Datetime, "ns"),
        RawTypes.TIMESTAMP_TZ: DatetimeTZ,
        RawTypes.BOOLEAN: Boolean,
        RawTypes.INTERVAL: Duration,
        RawTypes.TIME: Time,
        RawTypes.BLOB: Binary,
    }
)
