"""Data types Mapping for PQL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import timezone
from enum import Enum as PyEnum, StrEnum, auto
from functools import partial
from typing import Any, Literal

import duckdb
import pyochain as pc
from duckdb import sqltypes
from duckdb.sqltypes import DuckDBPyType


class RawTypes(StrEnum):
    LIST = auto()
    STRUCT = auto()
    ARRAY = auto()
    ENUM = auto()
    MAP = auto()
    DECIMAL = auto()
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


type Incomplete = Any
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
    def from_duckdb(  # noqa: PLR0911
        dtype: DuckDBPyType, deferred_time_zone: DeferredTimeZone
    ) -> DataType:
        match dtype.id:
            case RawTypes.LIST:
                return into_list(dtype, deferred_time_zone)
            case RawTypes.STRUCT:
                return into_struct(dtype, deferred_time_zone)
            case RawTypes.ARRAY:
                return into_array(dtype, deferred_time_zone)
            case RawTypes.ENUM:
                return into_enum(dtype)
            case RawTypes.TIMESTAMP_TZ:
                return Datetime(time_zone=deferred_time_zone.time_zone)
            case RawTypes.DECIMAL:
                return into_decimal(dtype)
            case _ as dtype_id:
                return (
                    NON_NESTED_MAP.get_item(dtype_id)
                    .map(lambda dt: dt())
                    .expect(f"Unsupported data type: {dtype_id}")
                )


def into_list(dtype: DuckDBPyType, deferred_time_zone: DeferredTimeZone) -> List:
    return List(DataType.from_duckdb(dtype.child, deferred_time_zone))


def into_struct(dtype: DuckDBPyType, deferred_time_zone: DeferredTimeZone) -> Struct:
    return (
        pc.Vec.from_ref(dtype.children)
        .iter()
        .map_star(
            lambda name, dtype: (
                name,
                DataType.from_duckdb(dtype, deferred_time_zone),  # pyright: ignore[reportArgumentType]
            )
        )
        .into(Struct)
    )


def into_array(dtype: DuckDBPyType, deferred_time_zone: DeferredTimeZone) -> Array:
    def _node(
        node: list[tuple[str, Any]],
    ) -> pc.Option[list[tuple[str, Any]]]:
        match node[0][1].id:
            case RawTypes.ARRAY:
                return pc.Some(node[0][1].children)
            case _:
                return pc.NONE

    def _first_or(shape: pc.Seq[int]) -> int | Iterable[int]:
        return shape.first() if shape.length() == 1 else shape

    levels = pc.Iter.successors(pc.Some(dtype.children), _node).collect()
    shape = (
        levels.iter()
        .map(lambda node: node[1][1])
        .collect()
        .rev()
        .collect()
        .into(_first_or)  # pyright: ignore[reportArgumentType]
    )
    return Array(
        DataType.from_duckdb(levels.last()[0][1], deferred_time_zone),  # pyright: ignore[reportArgumentType]
        shape=shape,
    )


def into_enum(dtype: DuckDBPyType) -> Enum:
    categories: Incomplete = dtype.children[0][1]
    return Enum(categories)


def into_decimal(dtype: DuckDBPyType) -> Decimal:
    precision: Incomplete
    scale: Incomplete
    (_, precision), (_, scale) = dtype.children
    return Decimal(precision, scale)


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
    time_zone: str | timezone | None = None

    def sql(self) -> DuckDBPyType:
        if self.time_zone is None:
            return PRECISION_MAP.get(self.time_unit, sqltypes.TIMESTAMP)
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
    shape: int | Iterable[int]

    def sql(self) -> DuckDBPyType:
        match self.shape:
            case int() as size:
                return duckdb.array_type(self.inner.sql(), size)
            case _:
                shapes = pc.Iter(self.shape).map(lambda item: f"[{item}]").join("")
                return DuckDBPyType(f"{self.inner.sql()}{shapes}")


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


@dataclass(slots=True)
class DeferredTimeZone:
    """Object which gets passed between `native_to_narwhals_dtype` calls.

    DuckDB stores the time zone in the connection, rather than in the dtypes, so
    this ensures that when calculating the schema of a dataframe with multiple
    timezone-aware columns, that the connection's time zone is only fetched once.

    Note: we cannot make the time zone a cached `DuckDBLazyFrame` property because
    the time zone can be modified after `DuckDBLazyFrame` creation:
    """

    _rel: duckdb.DuckDBPyRelation
    _cached_time_zone: pc.Option[str] = field(default_factory=lambda: pc.NONE)

    @property
    def time_zone(self) -> str:
        """Fetch relation time zone (if it wasn't calculated already)."""
        if self._cached_time_zone.is_none():
            self._cached_time_zone = pc.Some(_fetch_rel_time_zone(self._rel))
        return self._cached_time_zone.unwrap()


def _fetch_rel_time_zone(rel: duckdb.DuckDBPyRelation) -> str:
    tbl = "duckdb_settings()"
    qry = f"""--sql
        SELECT value
        FROM {tbl}
        WHERE name = 'TimeZone'
        """

    return pc.Option(rel.query(tbl, qry).fetchone()).unwrap()[0]


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
        RawTypes.BOOLEAN: Boolean,
        RawTypes.INTERVAL: Duration,
        RawTypes.TIME: Time,
        RawTypes.BLOB: Binary,
    }
)
