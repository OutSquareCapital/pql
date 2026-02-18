from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import NamedTuple, Self, cast

import duckdb
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
    BLOB = sqltypes.BLOB.id


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
    def from_duckdb(cls, raw: RawNamedType) -> Self:
        name, dtype = raw
        return cls(name, parse(dtype))


@dataclass(slots=True)
class DType:
    """Base class for all parsed DuckDB types."""

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        raise NotImplementedError


@dataclass(slots=True)
class ScalarType(DType):
    """Leaf scalar DuckDB type (no nested children)."""

    duckdb_id: str

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        return cls(dtype.id)


@dataclass(slots=True)
class DecimalType(DType):
    """DuckDB `DECIMAL` dtype."""

    precision: NamedInt
    scale: NamedInt

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        precision, scale = Cast.into_decimal(dtype.children)
        return cls(NamedInt(*precision), NamedInt(*scale))


@dataclass(slots=True)
class EnumType(DType):
    """DuckDB `ENUM` dtype."""

    child: NamedValues

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        (inner,) = Cast.into_enum(dtype.children)
        return cls(NamedValues(*inner))


@dataclass(slots=True)
class ListType(DType):
    """DuckDB `LIST` dtype."""

    child: Field

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        return cls(Field.from_duckdb(*Cast.into_list(dtype.children)))


@dataclass(slots=True)
class ArrayType(DType):
    """DuckDB `ARRAY` dtype with fixed size."""

    child: Field
    size: NamedInt

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        (child), (size) = Cast.into_array(dtype.children)
        return cls(Field.from_duckdb(child), NamedInt(*size))


@dataclass(slots=True)
class StructType(DType):
    """DuckDB `STRUCT` type."""

    fields: tuple[Field, ...]

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        return cls(
            pc.Vec.from_ref(Cast.into_struct(dtype.children))
            .iter()
            .map(Field.from_duckdb)
            .collect(tuple)
        )


@dataclass(slots=True)
class MapType(DType):
    """DuckDB `MAP` dtype."""

    key: Field
    value: Field

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        key, value = Cast.into_map(dtype.children)
        return cls(Field.from_duckdb(key), Field.from_duckdb(value))


@dataclass(slots=True)
class UnionType(DType):
    """DuckDB `UNION` dtype."""

    fields: tuple[Field, ...]

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> Self:
        return cls(
            pc.Vec.from_ref(Cast.into_union(dtype.children))
            .iter()
            .map(Field.from_duckdb)
            .collect(tuple)
        )


DTYPE_MAP: pc.Dict[str, type[DType]] = pc.Dict.from_ref(
    {
        RawTypes.LIST: ListType,
        RawTypes.ARRAY: ArrayType,
        RawTypes.STRUCT: StructType,
        RawTypes.MAP: MapType,
        RawTypes.UNION: UnionType,
        RawTypes.ENUM: EnumType,
        RawTypes.DECIMAL: DecimalType,
    }
)
"""Mapping of parsing strategies for each raw DuckDB type id.

If a type id is not present in this map, it will be parsed as a simple ScalarType."""


def parse(dtype: DuckDBPyType) -> DType:
    """Main entry point to convert a raw DuckDBPyType into a parsed DType.

    Recursively matches the raw type id to the appropriate parsing logic.
    """
    return DTYPE_MAP.get_item(dtype.id).unwrap_or(ScalarType).from_duckdb(dtype)


def _relation_types() -> duckdb.DuckDBPyRelation:
    qry = """--sql
    SELECT
        [1, 2]::INTEGER[] AS list_col,
        [[1, 2, 3], [4, 5, 6]]::INTEGER[2][3] AS array_col,
        {'name': 'alice', 'flags': [true, false], 'status': 'on'::ENUM('on', 'off')} AS struct_col,
        MAP([1, 2], ['x', 'y']) AS map_col,
        union_value(num := 2)::UNION(num INTEGER, txt VARCHAR) AS union_col,
        'on'::ENUM('on', 'off') AS enum_col,
        12.34::DECIMAL(10, 2) AS decimal_col
    """
    return duckdb.from_query(qry)


def main() -> None:
    rel = _relation_types()
    return (
        pc.Vec.from_ref(rel.columns)
        .iter()
        .zip(cast(list[DuckDBPyType], rel.dtypes), strict=True)
        .map_star(lambda col_name, duck_type: f"{col_name}: {parse(duck_type)}")
        .for_each(print)
    )


if __name__ == "__main__":
    main()
