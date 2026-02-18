from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import duckdb
import pyochain as pc
from duckdb.sqltypes import DuckDBPyType

type RawNamedType = tuple[str, DuckDBPyType]
type RawNamedInt = tuple[str, int]
type RawArrayChildren = tuple[RawNamedType, RawNamedInt]
type RawStructChildren = list[RawNamedType]
type RawMapChildren = tuple[RawNamedType, RawNamedType]
type RawEnumChildren = tuple[tuple[str, tuple[str, ...] | list[str]]]
type RawUnionChildren = list[RawNamedType]
type RawDecimalChildren = tuple[RawNamedInt, RawNamedInt]


@dataclass(slots=True)
class ScalarType:
    """Leaf scalar DuckDB type (no nested children)."""

    duckdb_id: str


@dataclass(slots=True)
class DecimalType:
    """DuckDB DECIMAL metadata."""

    precision_name: str
    precision: int
    scale_name: str
    scale: int


@dataclass(slots=True)
class EnumType:
    """DuckDB ENUM values."""

    values_name: str
    values: tuple[str, ...]


@dataclass(slots=True)
class ListType:
    """DuckDB LIST type."""

    inner: DType


@dataclass(slots=True)
class ArrayType:
    """DuckDB ARRAY type with fixed shape."""

    child_name: str
    size_name: str
    inner: DType
    shape: tuple[int, ...]


@dataclass(slots=True)
class StructField:
    """Named field of a DuckDB STRUCT."""

    name: str
    dtype: DType


@dataclass(slots=True)
class StructType:
    """DuckDB STRUCT type."""

    fields: tuple[StructField, ...]


@dataclass(slots=True)
class MapType:
    """DuckDB MAP type."""

    key_name: str
    key: DType
    value_name: str
    value: DType


@dataclass(slots=True)
class UnionMember:
    """Named member of a DuckDB UNION."""

    name: str
    dtype: DType


@dataclass(slots=True)
class UnionType:
    """DuckDB UNION type with tag and members."""

    tag_name: str
    tag_dtype: DType
    members: tuple[UnionMember, ...]


@dataclass(slots=True)
class UnknownType:
    """Fallback for unhandled DuckDB type ids."""

    duckdb_id: str


type DType = (
    ScalarType
    | DecimalType
    | EnumType
    | ListType
    | ArrayType
    | StructType
    | MapType
    | UnionType
    | UnknownType
)


@dataclass(slots=True)
class _ArrayLevel:
    child_name: str
    child_dtype: DuckDBPyType
    size_name: str
    size: int


def _into_array(dtype: DuckDBPyType) -> ArrayType:

    def _as_array_level(raw_children: RawArrayChildren) -> _ArrayLevel:
        (child_name, child_dtype), (size_name, size) = raw_children
        return _ArrayLevel(child_name, child_dtype, size_name, size)

    levels = pc.Iter.successors(
        pc.Some(_as_array_level(cast(RawArrayChildren, dtype.children))),
        lambda level: (
            pc.Some(_as_array_level(cast(RawArrayChildren, level.child_dtype.children)))
            if level.child_dtype.id == "array"
            else pc.NONE
        ),
    ).collect()
    shape = levels.iter().map(lambda level: level.size).collect().rev().collect(tuple)
    root = levels.first()
    return ArrayType(
        root.child_name,
        root.size_name,
        parse_duckdb_type(levels.last().child_dtype),
        shape,
    )


def _into_struct(dtype: DuckDBPyType) -> StructType:
    fields = (
        pc.Vec.from_ref(cast(RawStructChildren, dtype.children))
        .iter()
        .map_star(
            lambda field_name, field_dtype: StructField(
                field_name, parse_duckdb_type(field_dtype)
            )
        )
        .collect(tuple)
    )
    return StructType(fields)


def _into_map(dtype: DuckDBPyType) -> MapType:
    (key_name, key_dtype), (value_name, value_dtype) = cast(
        RawMapChildren, dtype.children
    )
    return MapType(
        key_name,
        parse_duckdb_type(key_dtype),
        value_name,
        parse_duckdb_type(value_dtype),
    )


def _into_enum(dtype: DuckDBPyType) -> EnumType:
    values_name, values = cast(RawEnumChildren, dtype.children)[0]
    return EnumType(values_name=values_name, values=pc.Iter(values).collect(tuple))


def _into_union(dtype: DuckDBPyType) -> UnionType:
    tag_name, tag_dtype = cast(RawUnionChildren, dtype.children)[0]
    members = (
        pc.Vec.from_ref(cast(RawUnionChildren, dtype.children))
        .iter()
        .skip(1)
        .map_star(
            lambda member_name, member_dtype: UnionMember(
                member_name, parse_duckdb_type(member_dtype)
            )
        )
        .collect(tuple)
    )
    return UnionType(tag_name, parse_duckdb_type(tag_dtype), members)


def _into_decimal(dtype: DuckDBPyType) -> DecimalType:
    (precision_name, precision), (scale_name, scale) = cast(
        RawDecimalChildren, dtype.children
    )
    return DecimalType(precision_name, precision, scale_name, scale)


def _children_or_none(dtype: DuckDBPyType) -> pc.Option[list[object]]:
    try:
        return pc.Some(cast(list[object], dtype.children))
    except duckdb.InvalidInputException:
        return pc.NONE


def parse_duckdb_type(dtype: DuckDBPyType) -> DType:  # noqa: PLR0911
    match dtype.id:
        case "list":
            return ListType(parse_duckdb_type(dtype.child))
        case "array":
            return _into_array(dtype)
        case "struct":
            return _into_struct(dtype)
        case "map":
            return _into_map(dtype)
        case "union":
            return _into_union(dtype)
        case "enum":
            return _into_enum(dtype)
        case "decimal":
            return _into_decimal(dtype)
        case duckdb_id:
            match _children_or_none(dtype):
                case pc.Some(children) if len(children) > 0:
                    return UnknownType(duckdb_id)
                case _:
                    return ScalarType(duckdb_id)


def describe_dtype(dtype: DType) -> str:  # noqa: PLR0911
    match dtype:
        case ScalarType(duckdb_id):
            return f"Scalar<{duckdb_id}>"
        case DecimalType(precision_name, precision, scale_name, scale):
            return f"Decimal({precision_name}={precision}, {scale_name}={scale})"
        case EnumType(values_name, values):
            return f"Enum({values_name}={values!r})"
        case ListType(inner=inner):
            return f"List<{describe_dtype(inner)}>"
        case ArrayType(child_name, size_name, inner, shape):
            return f"Array({child_name}={describe_dtype(inner)}, {size_name}={shape!r})"
        case StructType(fields):
            items = (
                pc.Iter(fields)
                .map(lambda f: f"{f.name}: {describe_dtype(f.dtype)}")
                .join(", ")
            )
            return f"Struct<{items}>"
        case MapType(key_name, key, value_name, value):
            return f"Map({key_name}={describe_dtype(key)}, {value_name}={describe_dtype(value)})"
        case UnionType(tag_name, tag_dtype, members):
            members_txt = (
                pc.Iter(members)
                .map(lambda m: f"{m.name}: {describe_dtype(m.dtype)}")
                .join(", ")
            )
            return f"Union({tag_name}={describe_dtype(tag_dtype)}, {members_txt})"
        case UnknownType(duckdb_id):
            return f"Unknown<{duckdb_id}>"


def _relation_types() -> pc.Dict[str, DuckDBPyType]:
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
    rel = duckdb.from_query(qry)
    return (
        pc.Vec.from_ref(rel.columns)
        .iter()
        .zip(cast(list[DuckDBPyType], rel.dtypes), strict=True)
        .collect(pc.Dict)
    )


def main() -> None:
    return (
        _relation_types()
        .items()
        .iter()
        .map_star(
            lambda col_name, duck_type: (
                f"{col_name}: {describe_dtype(parse_duckdb_type(duck_type))}"
            )
        )
        .for_each(print)
    )


if __name__ == "__main__":
    main()
