from __future__ import annotations

from enum import StrEnum, auto

import polars as pl
import pyochain as pc

from .._utils import Builtins, DateTime, Decimal, Pql, Typing


class DuckDbTypes(StrEnum):
    """DuckDB type names."""

    NULL = '"NULL"'
    AGGREGATE_STATE = "AGGREGATE_STATE<?>"
    ANY = "ANY"
    ARRAY = "ARRAY"
    BIGINT = "BIGINT"
    BIGNUM = "BIGNUM"
    BIT = "BIT"
    BLOB = "BLOB"
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    DECIMAL = "DECIMAL"
    DOUBLE = "DOUBLE"
    DOUBLE_3 = "DOUBLE[3]"
    DOUBLE_ANY = "DOUBLE[ANY]"
    FLOAT = "FLOAT"
    FLOAT_3 = "FLOAT[3]"
    FLOAT_ANY = "FLOAT[ANY]"
    HUGEINT = "HUGEINT"
    INTEGER = "INTEGER"
    INTERVAL = "INTERVAL"
    INVALID = "INVALID"
    JSON = "JSON"
    KEY = "K"
    LAMBDA = "LAMBDA"
    LIST = "LIST"
    MAP = "MAP"
    MAP_NULL_NULL = 'MAP("NULL", "NULL")'
    MAP_K_V = "MAP(K, V)"
    SMALLINT = "SMALLINT"
    STRUCT = "STRUCT"
    STRUCT_EMPTY = "STRUCT()"
    GENERIC = Typing.T
    POINTER = "POINTER"
    TABLE = "TABLE"
    TIME = "TIME"
    TIME_NS = "TIME_NS"
    TIME_WITH_TIME_ZONE = "TIME WITH TIME ZONE"
    TIMESTAMP = "TIMESTAMP"
    TIMESTAMP_WITH_TIME_ZONE = "TIMESTAMP WITH TIME ZONE"
    TIMESTAMP_NS = "TIMESTAMP_NS"
    TINYINT = "TINYINT"
    UBIGINT = "UBIGINT"
    UHUGEINT = "UHUGEINT"
    UINTEGER = "UINTEGER"
    UNION = "UNION"
    USMALLINT = "USMALLINT"
    UTINYINT = "UTINYINT"
    UUID = "UUID"
    V = "V"
    VARCHAR = "VARCHAR"
    VARIANT = "VARIANT"
    YMD = "STRUCT(year BIGINT, month BIGINT, day BIGINT)"
    YMD_QUOTED = 'STRUCT("year" BIGINT, "month" BIGINT, "day" BIGINT)'
    ANY_ARRAY = "ANY[]"
    BIG_INT_ARRAY = "BIGINT[]"
    BOOLEAN_ARRAY = "BOOLEAN[]"
    DATE_ARRAY = "DATE[]"
    DECIMAL_ARRAY = "DECIMAL[]"
    DOUBLE_ARRAY = "DOUBLE[]"
    FLOAT_ARRAY = "FLOAT[]"
    HUGEINT_ARRAY = "HUGEINT[]"
    INTEGER_ARRAY = "INTEGER[]"
    JSON_ARRAY = "JSON[]"
    K_ARRAY = "K[]"
    NULL_ARRAY = '"NULL"[]'
    SMALLINT_ARRAY = "SMALLINT[]"
    STRUCT_KV_ARRAY = 'STRUCT("key" K, "value" V)[]'
    STRUCT_ARRAY = "STRUCT[]"
    STRUCT_K_V_ARRAY = "STRUCT(K, V)[]"
    TIME_TZ_ARRAY = "TIME WITH TIME ZONE[]"
    TIMESTAMP_TZ_ARRAY = "TIMESTAMP WITH TIME ZONE[]"
    TIMESTAMP_ARRAY = "TIMESTAMP[]"
    TIME_ARRAY = "TIME[]"
    TINYINT_ARRAY = "TINYINT[]"
    GENERIC_ARRAY = "T[]"
    GENERIC_2D_ARRAY = "T[][]"
    UBIGINT_ARRAY = "UBIGINT[]"
    VARCHAR_ARRAY = "VARCHAR[]"
    VARCHAR_2D_ARRAY = "VARCHAR[][]"
    V_ARRAY = "V[]"

    def into_py(self) -> str:
        def _base_py_for_value(value: str) -> str:
            return (
                CONVERSION_MAP.get_item(value)
                .filter(lambda x: x != Typing.SELF.value)
                .unwrap_or("")
            )

        match self.value:
            case inner if inner.endswith("[]") and self not in GENERIC_CONTAINER:
                element_type = _base_py_for_value(inner.removesuffix("[]"))
                return Pql.SEQ_LITERAL.of_type(element_type)
            case arr if "[" in arr:
                return _base_py_for_value(arr.partition("[")[0])
            case _ as value:
                return _base_py_for_value(value)


GENERIC_CONTAINER = pc.Set(
    (DuckDbTypes.ANY_ARRAY, DuckDbTypes.GENERIC_ARRAY, DuckDbTypes.V_ARRAY)
)


DTYPES = pl.Enum(DuckDbTypes)


class FuncTypes(StrEnum):
    """Function types in DuckDB."""

    PRAGMA = auto()
    """Settings-like functions."""
    TABLE = auto()
    TABLE_MACRO = auto()
    AGGREGATE = auto()
    MACRO = auto()
    SCALAR = auto()

    @classmethod
    def unwanted(cls) -> set[FuncTypes]:
        return {cls.TABLE, cls.TABLE_MACRO, cls.PRAGMA}


FUNC_TYPES = pl.Enum(FuncTypes)


class Categories(StrEnum):
    """DuckDB function categories."""

    NULL = auto()
    AGGREGATE = auto()
    ARRAY = auto()
    BITSTRING = auto()
    BLOB = auto()
    DATE = auto()
    LAMBDA = auto()
    LIST = auto()
    NUMERIC = auto()
    REGEX = auto()
    STRING = auto()
    STRUCT = auto()
    TEXT_SIMILARITY = auto()
    TIMESTAMP = auto()
    VARIANT = auto()


CATEGORY_TYPES = pl.Enum(Categories)


class Stability(StrEnum):
    """DuckDB function stability categories."""

    CONSISTENT = "CONSISTENT"
    CONSISTENT_WITHIN_QUERY = "CONSISTENT_WITHIN_QUERY"
    VOLATILE = "VOLATILE"


class SchemaName(StrEnum):
    MAIN = auto()
    PG_CATALOG = auto()


CONVERSION_MAP: pc.Dict[str, str] = pc.Dict(
    {
        DuckDbTypes.VARCHAR: Builtins.STR.value,
        DuckDbTypes.INTEGER: Builtins.INT.value,
        DuckDbTypes.BIGINT: Builtins.INT.value,
        DuckDbTypes.SMALLINT: Builtins.INT.value,
        DuckDbTypes.TINYINT: Builtins.INT.value,
        DuckDbTypes.HUGEINT: Builtins.INT.value,
        DuckDbTypes.UINTEGER: Builtins.INT.value,
        DuckDbTypes.UBIGINT: Builtins.INT.value,
        DuckDbTypes.USMALLINT: Builtins.INT.value,
        DuckDbTypes.UTINYINT: Builtins.INT.value,
        DuckDbTypes.UHUGEINT: Builtins.INT.value,
        DuckDbTypes.DOUBLE: Builtins.FLOAT.value,
        DuckDbTypes.FLOAT: Builtins.FLOAT.value,
        DuckDbTypes.DECIMAL: Decimal.DECIMAL.value,
        DuckDbTypes.BOOLEAN: Builtins.BOOL.value,
        DuckDbTypes.DATE: DateTime.DATE.value,
        DuckDbTypes.TIME: DateTime.TIME.value,
        DuckDbTypes.TIMESTAMP: DateTime.DATETIME.value,
        DuckDbTypes.INTERVAL: DateTime.TIMEDELTA.value,
        DuckDbTypes.BLOB: Builtins.BYTES.into_union(
            Builtins.BYTEARRAY, Builtins.MEMORYVIEW
        ),
        DuckDbTypes.BIT: Builtins.BYTES.into_union(
            Builtins.BYTEARRAY, Builtins.MEMORYVIEW
        ),
        DuckDbTypes.UUID: Builtins.STR.value,
        DuckDbTypes.JSON: Builtins.STR.value,
        DuckDbTypes.ANY: Typing.SELF.value,
        DuckDbTypes.LIST: Pql.SEQ_LITERAL.value,
        DuckDbTypes.MAP: Builtins.DICT.value,
        DuckDbTypes.STRUCT: Builtins.DICT.value,
        DuckDbTypes.ARRAY: Pql.SEQ_LITERAL.value,
        DuckDbTypes.UNION: Typing.SELF.value,
        DuckDbTypes.NULL: Builtins.NONE.value,
    }
)
