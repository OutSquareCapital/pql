import keyword
from enum import StrEnum, auto

import polars as pl
import pyochain as pc


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


class FuncTypes(StrEnum):
    """Function types in DuckDB."""

    PRAGMA = auto()
    TABLE = auto()
    TABLE_MACRO = auto()
    AGGREGATE = auto()
    MACRO = auto()
    SCALAR = auto()


@schema
class TableSchema:
    """Schema for DuckDB functions table."""

    database_name = pl.String
    database_oid = pl.String
    schema_name = pl.String
    function_name = pl.String
    alias_of = pl.String()
    function_type = pl.Enum(FuncTypes)
    description = pl.String()
    comment = pl.String()
    tags = pl.List(pl.Struct({"key": pl.String, "value": pl.String}))
    return_type = pl.String()
    parameters = pl.List(pl.String)
    parameter_types = pl.List(pl.String)
    varargs = pl.String()
    macro_definition = pl.String()
    has_side_effects = pl.Boolean()
    internal = pl.Boolean()
    function_oid = pl.Int64()
    examples = pl.List(pl.String)
    stability = pl.String()
    categories = pl.List(pl.String)


class PyTypes(StrEnum):
    """Python type names for DuckDB type mapping."""

    STR = auto()
    BOOL = auto()
    INT = auto()
    DATE = auto()
    FLOAT = auto()
    BYTES = auto()
    TIME = auto()
    DATETIME = auto()
    TIMEDELTA = auto()
    LIST = "list[object]"
    DICT = "dict[object, object]"
    EXPR = "SqlExpr"


# DuckDB type -> Python type hint mapping
DUCKDB_TYPE_MAP = pc.Dict.from_kwargs(
    VARCHAR=PyTypes.STR,
    INTEGER=PyTypes.INT,
    BIGINT=PyTypes.INT,
    SMALLINT=PyTypes.INT,
    TINYINT=PyTypes.INT,
    HUGEINT=PyTypes.INT,
    UINTEGER=PyTypes.INT,
    UBIGINT=PyTypes.INT,
    USMALLINT=PyTypes.INT,
    UTINYINT=PyTypes.INT,
    UHUGEINT=PyTypes.INT,
    DOUBLE=PyTypes.FLOAT,
    FLOAT=PyTypes.FLOAT,
    REAL=PyTypes.FLOAT,
    DECIMAL=PyTypes.FLOAT,
    BOOLEAN=PyTypes.BOOL,
    DATE=PyTypes.DATE,
    TIME=PyTypes.TIME,
    TIMESTAMP=PyTypes.DATETIME,
    INTERVAL=PyTypes.TIMEDELTA,
    BLOB=PyTypes.BYTES,
    BIT=PyTypes.BYTES,
    UUID=PyTypes.STR,
    JSON=PyTypes.STR,
    ANY=PyTypes.EXPR,
    LIST=PyTypes.LIST,
    MAP=PyTypes.DICT,
    STRUCT=PyTypes.DICT,
    ARRAY=PyTypes.LIST,
    UNION=PyTypes.EXPR,
)


FN_CATEGORY = pc.Dict.from_kwargs(
    scalar="Scalar Functions",
    aggregate="Aggregate Functions",
    table="Table Functions",
    macro="Macro Functions",
)
"""Mapping of DuckDB function types to categories."""
KWORDS = pc.Set(keyword.kwlist)
"""Python reserved keywords that need renaming when generating function names."""
BUILTINS = pc.Set(dir(__builtins__))
AMBIGUOUS = pc.Set("l")
SHADOWERS = KWORDS.union(BUILTINS).union(AMBIGUOUS)
"""Python built-in names."""
SKIP_FUNCTIONS = pc.Set(
    {
        # Internal functions
        "~~",
        "!~~",
        "~~~",
        "!~~~",
        "^@",
        "@",
        "||",
        "**",
        "!",
        # Already exposed differently
        "main.if",
    }
)
"""Functions to skip (internal, deprecated, or not useful via Python API)."""
CATEGORY_PATTERNS: pc.Seq[tuple[str, str]] = pc.Seq(
    (
        ("list_", "List Functions"),
        ("array_", "Array Functions"),
        ("map_", "Map Functions"),
        ("struct_", "Struct Functions"),
        ("regexp_", "Regular Expression Functions"),
        ("string_", "Text Functions"),
        ("date_", "Date Functions"),
        ("time_", "Time Functions"),
        ("timestamp_", "Timestamp Functions"),
        ("enum_", "Enum Functions"),
        ("union_", "Union Functions"),
        ("json_", "JSON Functions"),
        ("to_", "Conversion Functions"),
        ("from_", "Conversion Functions"),
        ("is", "Predicate Functions"),
        ("bit_", "Bitwise Functions"),
    )
)
"""Patterns to categorize functions based on their name prefixes."""
