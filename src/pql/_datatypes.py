"""Data types Mapping for PQL."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, Field, dataclass, field, fields
from enum import Enum as PyEnum
from typing import Any, Literal, Self

import pyochain as pc

from .sql import (
    ArrayType,
    DecimalType,
    DType,
    EnumType,
    ListType,
    MapType,
    RawTypes,
    ScalarType,
    SqlType,
    StructType,
    UnionType,
)

TimeUnit = Literal["s", "ms", "us", "ns"]


class DataType(ABC):
    """Base class for data types."""

    raw: SqlType
    __slots__ = ()

    @staticmethod
    def __from_sql__(dtype: SqlType) -> DataType:
        """Recursively convert a raw SQL type to a PQL DataType using the strategy pattern.

        This is not meant to be called directly by the user, and is only used internally by the schema property in `LazyFrame`.
        """
        return (
            NESTED_MAP.get_item(dtype.type_id)
            .map(lambda constructor: constructor.__from_raw__(dtype))
            .unwrap_or_else(lambda: NON_NESTED_MAP.get_item(dtype.type_id).unwrap())
        )


@dataclass(slots=True)
class Binary(DataType):
    raw = ScalarType.BLOB


@dataclass(slots=True)
class Time(DataType):
    raw = ScalarType.TIME


@dataclass(slots=True)
class Duration(DataType):
    raw = ScalarType.INTERVAL


@dataclass(slots=True)
class Boolean(DataType):
    raw = ScalarType.BOOLEAN


@dataclass(slots=True)
class String(DataType):
    raw = ScalarType.VARCHAR


@dataclass(slots=True)
class Date(DataType):
    raw = ScalarType.DATE


@dataclass(slots=True)
class Float32(DataType):
    raw = ScalarType.FLOAT


@dataclass(slots=True)
class Float64(DataType):
    raw = ScalarType.DOUBLE


@dataclass(slots=True)
class Int8(DataType):
    raw = ScalarType.TINYINT


@dataclass(slots=True)
class Int16(DataType):
    raw = ScalarType.SMALLINT


@dataclass(slots=True)
class Int32(DataType):
    raw = ScalarType.INTEGER


@dataclass(slots=True)
class Int64(DataType):
    raw = ScalarType.BIGINT


@dataclass(slots=True)
class Int128(DataType):
    raw = ScalarType.HUGEINT


@dataclass(slots=True)
class UInt8(DataType):
    raw = ScalarType.UTINYINT


@dataclass(slots=True)
class UInt16(DataType):
    raw = ScalarType.USMALLINT


@dataclass(slots=True)
class UInt32(DataType):
    raw = ScalarType.UINTEGER


@dataclass(slots=True)
class UInt64(DataType):
    raw = ScalarType.UBIGINT


@dataclass(slots=True)
class UInt128(DataType):
    raw = ScalarType.UHUGEINT


@dataclass(slots=True)
class DatetimeTZ(DataType):
    raw = ScalarType.TIMESTAMP_TZ


@dataclass(slots=True, init=False)
class Datetime(DataType):
    raw: DType
    time_unit: TimeUnit

    def __init__(self, time_unit: TimeUnit = "ns") -> None:
        self.raw = PRECISION_MAP.get_item(time_unit).expect(
            f"Unsupported time unit: {time_unit}"
        )
        self.time_unit = time_unit


@dataclass(slots=True, init=False)
class ComplexDataType[T: SqlType](DataType):
    """Base class for complex data types."""

    raw: T

    @classmethod
    def __from_raw__(cls, raw: T) -> DataType:
        """Create a new instance of the complex data type from the raw SQL type.

        This is not meant to be called directly by the user, and is only used internally by the strategy pattern.

        Populate the dataclass fields with their default values by iterating over them, since we bypass the `__init__()` method.
        """

        def _set_attr(dataclass_field: Field[Any]) -> None:
            if dataclass_field.default is not MISSING:
                setattr(instance, dataclass_field.name, dataclass_field.default)
            elif dataclass_field.default_factory is not MISSING:
                setattr(
                    instance, dataclass_field.name, dataclass_field.default_factory()
                )

        instance = cls.__new__(cls)
        pc.Iter(fields(cls)).for_each(_set_attr)
        instance.raw = raw
        return instance


@dataclass(slots=True, init=False)
class Union(ComplexDataType[UnionType]):
    _fields: pc.Option[pc.Seq[DataType]] = field(default_factory=lambda: pc.NONE)

    def __init__(self, fields: Iterable[DataType]) -> None:
        self.raw = UnionType.new(
            pc.Iter(fields).iter().map(lambda f: f.raw.to_duckdb())
        )

    @property
    def fields(self) -> pc.Seq[DataType]:
        if self._fields.is_none():
            self._fields = pc.Some(
                self.raw.fields.iter()
                .map(lambda field: DataType.__from_sql__(field.dtype))
                .collect()
            )
        return self._fields.unwrap()


@dataclass(slots=True, init=False)
class Map(ComplexDataType[MapType]):
    _key: pc.Option[DataType] = field(default_factory=lambda: pc.NONE)
    _value: pc.Option[DataType] = field(default_factory=lambda: pc.NONE)

    def __init__(self, key: DataType, value: DataType) -> None:
        self.raw = MapType.new(key.raw.to_duckdb(), value.raw.to_duckdb())

    @property
    def key(self) -> DataType:
        if self._key.is_none():
            self._key = pc.Some(DataType.__from_sql__(self.raw.key.dtype))
        return self._key.unwrap()

    @property
    def value(self) -> DataType:
        if self._value.is_none():
            self._value = pc.Some(DataType.__from_sql__(self.raw.value.dtype))
        return self._value.unwrap()


@dataclass(slots=True, init=False)
class Decimal(ComplexDataType[DecimalType]):
    def __init__(self, precision: int = 18, scale: int = 0) -> None:
        self.raw = DecimalType.new(precision, scale)

    @property
    def precision(self) -> int:
        return self.raw.precision.value

    @property
    def scale(self) -> int:
        return self.raw.scale.value


@dataclass(slots=True, init=False)
class Array(ComplexDataType[ArrayType]):
    _inner: pc.Option[DataType] = field(default_factory=lambda: pc.NONE)

    def __init__(self, inner: DataType, shape: int = 1) -> None:
        self.raw = ArrayType.new(inner.raw.to_duckdb(), shape)

    def with_dim(self, shape: int) -> Self:
        """Add another level of nesting to the array."""
        return self.__class__(self, shape)

    @property
    def inner(self) -> DataType:
        if self._inner.is_none():
            self._inner = pc.Some(DataType.__from_sql__(self.raw.child.dtype))
        return self._inner.unwrap()

    @property
    def shape(self) -> int:
        return self.raw.size.value


@dataclass(slots=True, init=False)
class List(ComplexDataType[ListType]):
    def __init__(self, inner: DataType) -> None:
        self.raw = ListType.new(inner.raw.to_duckdb())

    @property
    def inner(self) -> DataType:
        return DataType.__from_sql__(self.raw.child.dtype)


@dataclass(slots=True, init=False)
class Struct(ComplexDataType[StructType]):
    def __init__(
        self, fields: Mapping[str, DataType] | Iterable[tuple[str, DataType]]
    ) -> None:
        self.raw = StructType.new(
            pc.Dict(fields)
            .items()
            .iter()
            .map_star(lambda name, col: (name, col.raw.to_duckdb()))
        )

    @property
    def fields(self) -> pc.Dict[str, DataType]:
        return (
            self.raw.fields.iter()
            .map(lambda field: (field.name, DataType.__from_sql__(field.dtype)))
            .collect(pc.Dict)
        )


@dataclass(slots=True, init=False)
class Enum(ComplexDataType[EnumType]):
    def __init__(self, categories: Iterable[str] | type[PyEnum]) -> None:
        self.raw = EnumType.new(categories)

    @property
    def categories(self) -> pc.Vec[str]:
        return self.raw.child.values


PRECISION_MAP: pc.Dict[TimeUnit, DType] = pc.Dict.from_ref(
    {
        "s": DType(str(RawTypes.TIMESTAMP_S), RawTypes.TIMESTAMP_S),
        "ms": DType(str(RawTypes.TIMESTAMP_MS), RawTypes.TIMESTAMP_MS),
        "us": DType(str(RawTypes.TIMESTAMP), RawTypes.TIMESTAMP),
        "ns": DType(str(RawTypes.TIMESTAMP_NS), RawTypes.TIMESTAMP_NS),
    }
)


NESTED_MAP: pc.Dict[str, type[ComplexDataType[Any]]] = pc.Dict.from_ref(
    {
        RawTypes.LIST: List,
        RawTypes.STRUCT: Struct,
        RawTypes.MAP: Map,
        RawTypes.UNION: Union,
        RawTypes.ARRAY: Array,
        RawTypes.ENUM: Enum,
        RawTypes.DECIMAL: Decimal,
    }
)

NON_NESTED_MAP: pc.Dict[str, DataType] = pc.Dict.from_ref(
    {
        RawTypes.HUGEINT: Int128(),
        RawTypes.BIGINT: Int64(),
        RawTypes.INTEGER: Int32(),
        RawTypes.SMALLINT: Int16(),
        RawTypes.TINYINT: Int8(),
        RawTypes.UHUGEINT: UInt128(),
        RawTypes.UBIGINT: UInt64(),
        RawTypes.UINTEGER: UInt32(),
        RawTypes.USMALLINT: UInt16(),
        RawTypes.UTINYINT: UInt8(),
        RawTypes.DOUBLE: Float64(),
        RawTypes.FLOAT: Float32(),
        RawTypes.VARCHAR: String(),
        RawTypes.DATE: Date(),
        RawTypes.TIMESTAMP_S: Datetime("s"),
        RawTypes.TIMESTAMP_MS: Datetime("ms"),
        RawTypes.TIMESTAMP: Datetime(),
        RawTypes.TIMESTAMP_NS: Datetime("ns"),
        RawTypes.TIMESTAMP_TZ: DatetimeTZ(),
        RawTypes.BOOLEAN: Boolean(),
        RawTypes.INTERVAL: Duration(),
        RawTypes.TIME: Time(),
        RawTypes.BLOB: Binary(),
    }
)
