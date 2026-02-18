from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import NamedTuple, Self, cast

import pyochain as pc
from duckdb import sqltypes
from duckdb.sqltypes import DuckDBPyType


class RawTypes(StrEnum):
    """Raw DuckDB type ids, including both scalar and nested types.

    This structure is here to help match the raw type ids from DuckDBPyType to the appropriate parsing logic.
    """

    LIST = auto()
    STRUCT = auto()
    ARRAY = auto()
    ENUM = auto()
    MAP = auto()
    DECIMAL = auto()
    UNION = auto()
    JSON = auto()
    HUGEINT = sqltypes.HUGEINT.id
    BIGINT = sqltypes.BIGINT.id
    INTEGER = sqltypes.INTEGER.id
    SMALLINT = sqltypes.SMALLINT.id
    TINYINT = sqltypes.TINYINT.id
    UHUGEINT = sqltypes.UHUGEINT.id
    UBIGINT = sqltypes.UBIGINT.id
    UINTEGER = sqltypes.UINTEGER.id
    USMALLINT = sqltypes.USMALLINT.id
    UTINYINT = sqltypes.UTINYINT.id
    DOUBLE = sqltypes.DOUBLE.id
    FLOAT = sqltypes.FLOAT.id
    VARCHAR = sqltypes.VARCHAR.id
    DATE = sqltypes.DATE.id
    TIMESTAMP_S = sqltypes.TIMESTAMP_S.id
    TIMESTAMP_MS = sqltypes.TIMESTAMP_MS.id
    TIMESTAMP = sqltypes.TIMESTAMP.id
    TIMESTAMP_NS = sqltypes.TIMESTAMP_NS.id
    TIMESTAMP_TZ = sqltypes.TIMESTAMP_TZ.id
    BOOLEAN = sqltypes.BOOLEAN.id
    INTERVAL = sqltypes.INTERVAL.id
    TIME = sqltypes.TIME.id
    TIME_TZ = sqltypes.TIME_TZ.id
    BLOB = sqltypes.BLOB.id
    BIT = sqltypes.BIT.id
    UUID = sqltypes.UUID.id
    BIGNUM = auto()


# Raw type aliases for the unparsed children of each DuckDB type, used in the Cast namespace to convert from the raw DuckDBPyType.children to more specific structures for each type.
# Note that we type container of one element as tuples when they are in fact at runtime lists.
# This makes the unpacking more convenient.
type RawNamedType = tuple[str, DuckDBPyType]
type RawNamedInt = tuple[str, int]
type RawListChildren = tuple[RawNamedType]
type RawArrayChildren = tuple[RawNamedType, RawNamedInt]
type RawStructChildren = list[RawNamedType]
type RawMapChildren = tuple[RawNamedType, RawNamedType]
type RawEnumChildren = tuple[tuple[str, list[str]]]
type RawUnionChildren = list[RawNamedType]
type RawDecimalChildren = tuple[RawNamedInt, RawNamedInt]


class Cast:
    """Namespace for unsafe casts from raw DuckDBPyType children to more specific types.

    Note that these casts don't have any runtime effect, and solely act as a first step to convert the raw values more conviently to concrete types.
    """

    @staticmethod
    def into_list(dtype: object) -> RawListChildren:
        return cast(RawListChildren, dtype)

    @staticmethod
    def into_array(dtype: object) -> RawArrayChildren:
        return cast(RawArrayChildren, dtype)

    @staticmethod
    def into_struct(dtype: object) -> RawStructChildren:
        return cast(RawStructChildren, dtype)

    @staticmethod
    def into_map(dtype: object) -> RawMapChildren:
        return cast(RawMapChildren, dtype)

    @staticmethod
    def into_enum(dtype: object) -> RawEnumChildren:
        return cast(RawEnumChildren, dtype)

    @staticmethod
    def into_union(dtype: object) -> RawUnionChildren:
        return cast(RawUnionChildren, dtype)

    @staticmethod
    def into_decimal(dtype: object) -> RawDecimalChildren:
        return cast(RawDecimalChildren, dtype)


class NamedInt(NamedTuple):
    """Named integer value from DuckDB type children, used for things like array sizes and decimal precision/scale."""

    name: str
    value: int


class NamedValues(NamedTuple):
    """Named string list from DuckDB type children, used for enum values."""

    name: str
    values: list[str]


@dataclass(slots=True)
class Field:
    """Named parsed field."""

    name: str
    dtype: DType

    @classmethod
    def from_raw(cls, raw: RawNamedType) -> Self:
        name, dtype = raw
        return cls(name, parse(dtype))


@dataclass(slots=True)
class DType:
    """Base class for all parsed DuckDB types."""

    physical: str
    type_id: str

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        return cls(str(dtype), dtype.id)


@dataclass(slots=True)
class VarcharType(DType):
    """DuckDB `VARCHAR` dtype, which can also represent `JSON` when the physical type is `VARCHAR` but the logical type is `JSON`."""

    logical: str

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        match str(dtype):
            case RawTypes.JSON:
                return cls(str(dtype), dtype.id, RawTypes.JSON)
            case _:
                return cls(str(dtype), dtype.id, dtype.id)


@dataclass(slots=True)
class DecimalType(DType):
    """DuckDB `DECIMAL` dtype."""

    precision: NamedInt
    scale: NamedInt

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        precision, scale = Cast.into_decimal(dtype.children)
        return cls(str(dtype), dtype.id, NamedInt(*precision), NamedInt(*scale))


@dataclass(slots=True)
class EnumType(DType):
    """DuckDB `ENUM` dtype."""

    child: NamedValues

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        (inner,) = Cast.into_enum(dtype.children)
        return cls(str(dtype), dtype.id, NamedValues(*inner))


@dataclass(slots=True)
class ListType(DType):
    """DuckDB `LIST` dtype."""

    child: Field

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        return cls(
            str(dtype), dtype.id, Field.from_raw(*Cast.into_list(dtype.children))
        )


@dataclass(slots=True)
class ArrayType(DType):
    """DuckDB `ARRAY` dtype with fixed size."""

    child: Field
    size: NamedInt

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        child, size = Cast.into_array(dtype.children)
        return cls(str(dtype), dtype.id, Field.from_raw(child), NamedInt(*size))


@dataclass(slots=True)
class StructType(DType):
    """DuckDB `STRUCT` type."""

    fields: tuple[Field, ...]

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        return cls(
            str(dtype),
            dtype.id,
            pc.Vec.from_ref(Cast.into_struct(dtype.children))
            .iter()
            .map(Field.from_raw)
            .collect(tuple),
        )


@dataclass(slots=True)
class MapType(DType):
    """DuckDB `MAP` dtype."""

    key: Field
    value: Field

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        key, value = Cast.into_map(dtype.children)
        return cls(str(dtype), dtype.id, Field.from_raw(key), Field.from_raw(value))


@dataclass(slots=True)
class UnionType(DType):
    """DuckDB `UNION` dtype."""

    fields: tuple[Field, ...]

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        return cls(
            str(dtype),
            dtype.id,
            pc.Vec.from_ref(Cast.into_union(dtype.children))
            .iter()
            .map(Field.from_raw)
            .collect(tuple),
        )


type NestedType = ListType | ArrayType | StructType | UnionType
"""Types who can have nested children, and thus require recursive parsing logic."""
type ConfiguredType = DecimalType | EnumType | VarcharType
"""Types who are not nested, but have additional configuration parameters (like precision/scale for `Decimal`, or logical type for `Varchar`)."""
type SqlType = NestedType | ConfiguredType | DType
"""All possible parsed types, including simple scalar types which are represented as the base `DType`."""
DTYPE_MAP: pc.Dict[str, type[SqlType]] = pc.Dict.from_ref(
    {
        RawTypes.LIST: ListType,
        RawTypes.ARRAY: ArrayType,
        RawTypes.STRUCT: StructType,
        RawTypes.MAP: MapType,
        RawTypes.UNION: UnionType,
        RawTypes.ENUM: EnumType,
        RawTypes.DECIMAL: DecimalType,
        RawTypes.VARCHAR: VarcharType,
    }
)
"""Mapping of parsing strategies for each raw `DuckDB` type id.

If a type id is not present in this map, it will be parsed as a simple `DType`."""


def parse(dtype: DuckDBPyType) -> SqlType:
    """Main entry point to convert a raw DuckDBPyType into a parsed `DType`.

    Recursively matches the raw type id to the appropriate parsing logic.
    """
    return DTYPE_MAP.get_item(dtype.id).unwrap_or(DType).from_duckdb(dtype)
