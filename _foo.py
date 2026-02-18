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
type ParsedType = NestedType | ConfiguredType | DType
"""All possible parsed types, including simple scalar types which are represented as the base `DType`."""
DTYPE_MAP: pc.Dict[str, type[ParsedType]] = pc.Dict.from_ref(
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


def parse(dtype: DuckDBPyType) -> ParsedType:
    """Main entry point to convert a raw DuckDBPyType into a parsed `DType`.

    Recursively matches the raw type id to the appropriate parsing logic.
    """
    return DTYPE_MAP.get_item(dtype.id).unwrap_or(DType).from_duckdb(dtype)


def _relation_types() -> duckdb.DuckDBPyRelation:
    qry = """--sql
    SELECT
        -- Scalar types
        true::BOOLEAN AS bool_col,
        127::TINYINT AS tinyint_col,
        32767::SMALLINT AS smallint_col,
        2147483647::INTEGER AS int_col,
        9223372036854775807::BIGINT AS bigint_col,
        170141183460469231731687303715884105727::HUGEINT AS hugeint_col,
        255::UTINYINT AS utinyint_col,
        65535::USMALLINT AS usmallint_col,
        4294967295::UINTEGER AS uinteger_col,
        18446744073709551615::UBIGINT AS ubigint_col,
        340282366920938463463374607431768211455::UHUGEINT AS uhugeint_col,
        3.14::FLOAT AS float_col,
        2.718281828::DOUBLE AS double_col,
        'hello world'::VARCHAR AS varchar_col,
        '2025-02-18'::DATE AS date_col,
        '14:30:45'::TIME AS time_col,
        '2025-02-18 14:30:45'::TIMESTAMP AS timestamp_col,
        '2025-02-18T14:30:45'::TIMESTAMP_S AS timestamp_s_col,
        '2025-02-18T14:30:45.123'::TIMESTAMP_MS AS timestamp_ms_col,
        '2025-02-18T14:30:45.123456789'::TIMESTAMP_NS AS timestamp_ns_col,
        '2025-02-18 14:30:45-05:00'::TIMESTAMPTZ AS timestamptz_col,
        INTERVAL '1 days 2 hours 30 minutes' AS interval_col,
        x'48656c6c6f'::BLOB AS blob_col,
        -- Nested types
        [1, 2, 3]::INTEGER[] AS list_int_col,
        ['a', 'b', 'c']::VARCHAR[] AS list_varchar_col,
        [[1, 2], [3, 4], [5, 6]]::INTEGER[3][2] AS array_2d_col,
        [['x', 'y'], ['z', 'w']]::VARCHAR[2][2] AS array_varchar_col,
        {'id': 1, 'name': 'alice', 'active': true} AS struct_simple_col,
        {'name': 'bob', 'tags': ['rust', 'python'], 'score': 95.5} AS struct_nested_col,
        MAP([1, 2, 3], ['one', 'two', 'three']) AS map_int_varchar_col,
        MAP(['a', 'b'], [10, 20]) AS map_varchar_int_col,
        union_value(num := 42)::UNION(num INTEGER, txt VARCHAR) AS union_num_col,
        union_value(txt := 'hello')::UNION(num INTEGER, txt VARCHAR) AS union_txt_col,
        'on'::ENUM('on', 'off', 'pending') AS enum_status_col,
        'medium'::ENUM('small', 'medium', 'large') AS enum_size_col,
        99.99::DECIMAL(10, 2) AS decimal_price_col,
        123456789.123456::DECIMAL(15, 6) AS decimal_precision_col,
        [{'x': 1}, {'x': 2}]::STRUCT(x INTEGER)[] AS list_struct_col,
        {'coords': [1.5, 2.5, 3.5], 'metadata': {'type': 'point'}} AS struct_complex_col,
        '101010'::BIT AS bit_col,
        gen_random_uuid() AS uuid_col,
        '14:30:45+05:00'::TIMETZ AS timetz_col,
        123456789012345678901234567890::BIGNUM AS bignum_col,
        '{"name": "alice", "age": 30}'::JSON AS json_col,
    """
    return duckdb.from_query(qry)


def main() -> None:
    rel = _relation_types()
    return (
        pc.Vec.from_ref(rel.columns)
        .iter()
        .zip(cast(list[DuckDBPyType], rel.dtypes), strict=True)
        .map_star(lambda col_name, duck_type: (col_name, parse(duck_type)))
        .map_star(
            lambda col_name, parsed_type: {
                "column": col_name,
                "parsed_type": parsed_type,
            }
        )
        .for_each(print)
    )


if __name__ == "__main__":
    main()
