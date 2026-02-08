from dataclasses import asdict, dataclass, field
from enum import StrEnum, auto

import polars as pl
import pyochain as pc


class SchemaName(StrEnum):
    MAIN = auto()
    PG_CATALOG = auto()


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
    GENERIC = "T"
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


DTYPES = pl.Enum(DuckDbTypes)


class FuncTypes(StrEnum):
    """Function types in DuckDB."""

    PRAGMA = auto()
    TABLE = auto()
    TABLE_MACRO = auto()
    AGGREGATE = auto()
    MACRO = auto()
    SCALAR = auto()


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


def schema(cls: type[object]) -> pl.Schema:
    """Simple decorator for creating Polars Schema from class attributes."""

    def _is_polars_dtype(_k: str, v: object) -> bool:
        return isinstance(v, (type, pl.DataType)) and (
            isinstance(v, pl.DataType) or issubclass(v, pl.DataType)
        )

    return (
        pc.Dict.from_object(cls)
        .items()
        .iter()
        .filter_star(_is_polars_dtype)
        .collect(pl.Schema)
    )


@schema
class TableSchema:
    """Schema for DuckDB functions table."""

    database_name = pl.String
    database_oid = pl.UInt8
    schema_name = pl.Enum(SchemaName)
    function_name = pl.String
    alias_of = pl.String()
    function_type = FUNC_TYPES
    description = pl.String()
    """Only present for scalar and aggregate functions."""
    comment = pl.String()
    tags = pl.List(pl.Struct({"key": pl.Null(), "value": pl.Null()}))
    return_type = DTYPES
    parameters = pl.List(pl.String)
    parameter_types = pl.List(DTYPES)
    varargs = DTYPES
    macro_definition = pl.String()
    has_side_effects = pl.Boolean()
    internal = pl.Boolean
    function_oid = pl.UInt16
    examples = pl.List(pl.String)
    stability = pl.Enum(Stability)
    categories = pl.List(CATEGORY_TYPES)


@dataclass(slots=True)
class ParamLens:
    sig_param_count: pl.Expr = field(default=pl.col("sig_param_count"))
    min_params_per_fn: pl.Expr = field(default=pl.col("min_params_per_fn"))
    min_params_per_fn_cat_desc: pl.Expr = field(
        default=pl.col("min_params_per_fn_cat_desc")
    )


@dataclass(slots=True)
class PyCols:
    namespace: pl.Expr = field(default=pl.col("namespace"))
    name: pl.Expr = field(default=pl.col("py_name"))
    types: pl.Expr = field(default=pl.col("py_types"))


@dataclass(slots=True)
class Params:
    names: pl.Expr = field(default=pl.col("param_names"))
    idx: pl.Expr = field(default=pl.col("param_idx"))
    lens: ParamLens = field(default_factory=ParamLens)


@dataclass(slots=True)
class ParamLists:
    signatures: pl.Expr = field(default=pl.col("param_sig_list"))
    docs: pl.Expr = field(default=pl.col("param_doc_join"))
    names: pl.Expr = field(default=pl.col("param_names_join"))


@dataclass(slots=True)
class DuckCols:
    function_name: pl.Expr = field(default=pl.col("function_name"))
    function_type: pl.Expr = field(default=pl.col("function_type"))
    description: pl.Expr = field(default=pl.col("description"))
    categories: pl.Expr = field(default=pl.col("categories"))
    varargs: pl.Expr = field(default=pl.col("varargs"))
    alias_of: pl.Expr = field(default=pl.col("alias_of"))
    parameters: pl.Expr = field(default=pl.col("parameters"))
    parameter_types: pl.Expr = field(default=pl.col("parameter_types"))

    def to_dict(self) -> pc.Dict[str, pl.Expr]:

        return pc.Dict.from_ref(asdict(self))
