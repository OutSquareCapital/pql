"""Data types Mapping for PQL."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from dataclasses import MISSING, Field, dataclass, field, fields
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Any, Self

import pyochain as pc

from . import sql

if TYPE_CHECKING:
    from ._typing import TimeUnit
    from .sql.typing import IntoDict


class DataType(ABC):
    """Base class for data types."""

    raw: sql.SqlType
    __slots__ = ()

    @staticmethod
    def __from_sql__(dtype: sql.SqlType) -> DataType:
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
    raw = sql.ScalarType.BLOB


@dataclass(slots=True)
class Time(DataType):
    raw = sql.ScalarType.TIME


@dataclass(slots=True)
class Duration(DataType):
    raw = sql.ScalarType.INTERVAL


@dataclass(slots=True)
class Boolean(DataType):
    raw = sql.ScalarType.BOOLEAN


@dataclass(slots=True)
class String(DataType):
    raw = sql.ScalarType.VARCHAR


@dataclass(slots=True)
class Date(DataType):
    raw = sql.ScalarType.DATE


@dataclass(slots=True)
class Float32(DataType):
    raw = sql.ScalarType.FLOAT


@dataclass(slots=True)
class Float64(DataType):
    raw = sql.ScalarType.DOUBLE


@dataclass(slots=True)
class Int8(DataType):
    raw = sql.ScalarType.TINYINT


@dataclass(slots=True)
class Int16(DataType):
    raw = sql.ScalarType.SMALLINT


@dataclass(slots=True)
class Int32(DataType):
    raw = sql.ScalarType.INTEGER


@dataclass(slots=True)
class Int64(DataType):
    raw = sql.ScalarType.BIGINT


@dataclass(slots=True)
class Int128(DataType):
    raw = sql.ScalarType.HUGEINT


@dataclass(slots=True)
class UInt8(DataType):
    raw = sql.ScalarType.UTINYINT


@dataclass(slots=True)
class UInt16(DataType):
    raw = sql.ScalarType.USMALLINT


@dataclass(slots=True)
class UInt32(DataType):
    raw = sql.ScalarType.UINTEGER


@dataclass(slots=True)
class UInt64(DataType):
    raw = sql.ScalarType.UBIGINT


@dataclass(slots=True)
class UInt128(DataType):
    raw = sql.ScalarType.UHUGEINT


@dataclass(slots=True)
class DatetimeTZ(DataType):
    raw = sql.ScalarType.TIMESTAMP_TZ


@dataclass(slots=True, init=False)
class Datetime(DataType):
    raw: sql.DType
    time_unit: TimeUnit

    def __init__(self, time_unit: TimeUnit = "ns") -> None:
        self.raw = PRECISION_MAP.get_item(time_unit).expect(
            f"Unsupported time unit: {time_unit}"
        )
        self.time_unit = time_unit


@dataclass(slots=True, init=False)
class ComplexDataType[T: sql.SqlType](DataType):
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
class Union(ComplexDataType[sql.UnionType]):
    _fields: pc.Option[pc.Seq[DataType]] = field(default_factory=lambda: pc.NONE)

    def __init__(self, fields: Iterable[DataType]) -> None:
        self.raw = sql.UnionType.new(
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
class Map(ComplexDataType[sql.MapType]):
    _key: pc.Option[DataType] = field(default_factory=lambda: pc.NONE)
    _value: pc.Option[DataType] = field(default_factory=lambda: pc.NONE)

    def __init__(self, key: DataType, value: DataType) -> None:
        self.raw = sql.MapType.new(key.raw.to_duckdb(), value.raw.to_duckdb())

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
class Decimal(ComplexDataType[sql.DecimalType]):
    def __init__(self, precision: int = 18, scale: int = 0) -> None:
        self.raw = sql.DecimalType.new(precision, scale)

    @property
    def precision(self) -> int:
        return self.raw.precision.value

    @property
    def scale(self) -> int:
        return self.raw.scale.value


@dataclass(slots=True, init=False)
class Array(ComplexDataType[sql.ArrayType]):
    _inner: pc.Option[DataType] = field(default_factory=lambda: pc.NONE)

    def __init__(self, inner: DataType, size: int = 1) -> None:
        self.raw = sql.ArrayType.new(inner.raw.to_duckdb(), size)

    def with_dim(self, size: int) -> Self:
        """Add another level of nesting to the array."""
        return self.__class__(self, size)

    @property
    def inner(self) -> DataType:
        if self._inner.is_none():
            self._inner = pc.Some(DataType.__from_sql__(self.raw.child.dtype))
        return self._inner.unwrap()

    @property
    def shape(self) -> int:
        return self.raw.size.value


@dataclass(slots=True, init=False)
class List(ComplexDataType[sql.ListType]):
    _inner: pc.Option[DataType] = field(default_factory=lambda: pc.NONE)

    def __init__(self, inner: DataType) -> None:
        self.raw = sql.ListType.new(inner.raw.to_duckdb())

    @property
    def inner(self) -> DataType:
        if self._inner.is_none():
            self._inner = pc.Some(DataType.__from_sql__(self.raw.child.dtype))
        return self._inner.unwrap()


@dataclass(slots=True, init=False)
class Struct(ComplexDataType[sql.StructType]):
    _fields: pc.Option[pc.Dict[str, DataType]] = field(default_factory=lambda: pc.NONE)

    def __init__(self, fields: IntoDict[str, DataType]) -> None:
        self.raw = sql.StructType.new(
            pc.Dict(fields)
            .items()
            .iter()
            .map_star(lambda name, col: (name, col.raw.to_duckdb()))
        )

    @property
    def fields(self) -> pc.Dict[str, DataType]:
        if self._fields.is_none():
            self._fields = pc.Some(
                self.raw.fields.iter()
                .map(lambda field: (field.name, DataType.__from_sql__(field.dtype)))
                .collect(pc.Dict)
            )
        return self._fields.unwrap()


@dataclass(slots=True, init=False)
class Enum(ComplexDataType[sql.EnumType]):
    def __init__(self, categories: Iterable[str] | type[PyEnum]) -> None:
        self.raw = sql.EnumType.new(categories)

    @property
    def categories(self) -> pc.Vec[str]:
        return self.raw.child.values


PRECISION_MAP: pc.Dict[TimeUnit, sql.DType] = pc.Dict.from_ref(
    {
        "s": sql.ScalarType.TIMESTAMP_S,
        "ms": sql.ScalarType.TIMESTAMP_MS,
        "us": sql.ScalarType.TIMESTAMP,
        "ns": sql.ScalarType.TIMESTAMP_NS,
    }
)


NESTED_MAP: pc.Dict[str, type[ComplexDataType[Any]]] = pc.Dict.from_ref(
    {
        sql.RawTypes.LIST: List,
        sql.RawTypes.STRUCT: Struct,
        sql.RawTypes.MAP: Map,
        sql.RawTypes.UNION: Union,
        sql.RawTypes.ARRAY: Array,
        sql.RawTypes.ENUM: Enum,
        sql.RawTypes.DECIMAL: Decimal,
    }
)

NON_NESTED_MAP: pc.Dict[str, DataType] = pc.Dict.from_ref(
    {
        sql.RawTypes.HUGEINT: Int128(),
        sql.RawTypes.BIGINT: Int64(),
        sql.RawTypes.INTEGER: Int32(),
        sql.RawTypes.SMALLINT: Int16(),
        sql.RawTypes.TINYINT: Int8(),
        sql.RawTypes.UHUGEINT: UInt128(),
        sql.RawTypes.UBIGINT: UInt64(),
        sql.RawTypes.UINTEGER: UInt32(),
        sql.RawTypes.USMALLINT: UInt16(),
        sql.RawTypes.UTINYINT: UInt8(),
        sql.RawTypes.DOUBLE: Float64(),
        sql.RawTypes.FLOAT: Float32(),
        sql.RawTypes.VARCHAR: String(),
        sql.RawTypes.DATE: Date(),
        sql.RawTypes.TIMESTAMP_S: Datetime("s"),
        sql.RawTypes.TIMESTAMP_MS: Datetime("ms"),
        sql.RawTypes.TIMESTAMP: Datetime(),
        sql.RawTypes.TIMESTAMP_NS: Datetime("ns"),
        sql.RawTypes.TIMESTAMP_TZ: DatetimeTZ(),
        sql.RawTypes.BOOLEAN: Boolean(),
        sql.RawTypes.INTERVAL: Duration(),
        sql.RawTypes.TIME: Time(),
        sql.RawTypes.BLOB: Binary(),
    }
)
