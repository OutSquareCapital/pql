import builtins
import keyword
from abc import ABC, abstractmethod
from dataclasses import dataclass
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


FUNC_TYPES = pl.Enum(FuncTypes)


@schema
class TableSchema:
    """Schema for DuckDB functions table."""

    database_name = pl.String
    database_oid = pl.String
    schema_name = pl.String
    function_name = pl.String
    alias_of = pl.String()
    function_type = FUNC_TYPES
    description = pl.String()
    """Only present for scalar and aggregate functions."""
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
    DECIMAL = "Decimal"
    BYTES = auto()
    TIME = auto()
    DATETIME = auto()
    TIMEDELTA = auto()
    LIST = "list[object]"
    DICT = "dict[object, object]"
    EXPR = "SqlExpr"


class DuckDbTypes(StrEnum):
    """DuckDB type names."""

    VARCHAR = auto()
    INTEGER = auto()
    BIGINT = auto()
    SMALLINT = auto()
    TINYINT = auto()
    HUGEINT = auto()
    UINTEGER = auto()
    UBIGINT = auto()
    USMALLINT = auto()
    UTINYINT = auto()
    UHUGEINT = auto()
    DOUBLE = auto()
    FLOAT = auto()
    REAL = auto()
    DECIMAL = auto()
    BOOLEAN = auto()
    DATE = auto()
    TIME = auto()
    TIMESTAMP = auto()
    INTERVAL = auto()
    BLOB = auto()
    BIT = auto()
    UUID = auto()
    JSON = auto()
    ANY = auto()
    LIST = auto()
    MAP = auto()
    STRUCT = auto()
    ARRAY = auto()
    UNION = auto()


CONVERSION_MAP: pc.Dict[DuckDbTypes, PyTypes] = pc.Dict(
    {
        DuckDbTypes.VARCHAR: PyTypes.STR,
        DuckDbTypes.INTEGER: PyTypes.INT,
        DuckDbTypes.BIGINT: PyTypes.INT,
        DuckDbTypes.SMALLINT: PyTypes.INT,
        DuckDbTypes.TINYINT: PyTypes.INT,
        DuckDbTypes.HUGEINT: PyTypes.INT,
        DuckDbTypes.UINTEGER: PyTypes.INT,
        DuckDbTypes.UBIGINT: PyTypes.INT,
        DuckDbTypes.USMALLINT: PyTypes.INT,
        DuckDbTypes.UTINYINT: PyTypes.INT,
        DuckDbTypes.UHUGEINT: PyTypes.INT,
        DuckDbTypes.DOUBLE: PyTypes.FLOAT,
        DuckDbTypes.FLOAT: PyTypes.FLOAT,
        DuckDbTypes.REAL: PyTypes.FLOAT,
        DuckDbTypes.DECIMAL: PyTypes.DECIMAL,
        DuckDbTypes.BOOLEAN: PyTypes.BOOL,
        DuckDbTypes.DATE: PyTypes.DATE,
        DuckDbTypes.TIME: PyTypes.TIME,
        DuckDbTypes.TIMESTAMP: PyTypes.DATETIME,
        DuckDbTypes.INTERVAL: PyTypes.TIMEDELTA,
        DuckDbTypes.BLOB: PyTypes.BYTES,
        DuckDbTypes.BIT: PyTypes.BYTES,
        DuckDbTypes.UUID: PyTypes.STR,
        DuckDbTypes.JSON: PyTypes.STR,
        DuckDbTypes.ANY: PyTypes.EXPR,
        DuckDbTypes.LIST: PyTypes.LIST,
        DuckDbTypes.MAP: PyTypes.DICT,
        DuckDbTypes.STRUCT: PyTypes.DICT,
        DuckDbTypes.ARRAY: PyTypes.LIST,
        DuckDbTypes.UNION: PyTypes.EXPR,
    }
)
"""DuckDB type -> Python type hint mapping."""

KWORDS = pc.Set(keyword.kwlist)
"""Python reserved keywords that need renaming when generating function names."""
SHADOWERS = KWORDS.union(pc.Set(dir(builtins))).union(
    pc.Set(("l", "date", "time", "datetime", "timedelta"))
)
"""Names that should be renamed to avoid shadowing."""
OPERATOR_MAP = pc.Set(
    {
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "**",
        "&",
        "|",
        "^",
        "~",
        "&&",
        "||",
        "@",
        "^@",
        "@>",
        "<@",
        "<->",
        "<=>",
        "<<",
        ">>",
        "->>",
        "~~",
        "!~~",
        "~~*",
        "!~~*",
        "~~~",
        "!__postfix",
        "!",
    }
)
"""Mapping of SQL operators to Python function names."""


@dataclass(slots=True)
class CatRule[T](ABC):
    pattern: T
    label: str

    @abstractmethod
    def into_expr(self, expr: pl.Expr) -> pl.Expr: ...


class Prefix(CatRule[str]):
    def into_expr(self, expr: pl.Expr) -> pl.Expr:
        return pl.when(expr.str.starts_with(self.pattern)).then(pl.lit(self.label))


class TypeRule(CatRule[FuncTypes]):
    def into_expr(self, expr: pl.Expr) -> pl.Expr:  # noqa: ARG002
        return pl.when(
            pl.col("function_type").eq(pl.lit(self.pattern, dtype=FUNC_TYPES))
        ).then(pl.lit(self.label))


CATEGORY_RULES = pc.Seq(
    (
        Prefix("list", "List"),
        Prefix("array", "Array"),
        Prefix("map", "Map"),
        Prefix("struct", "Struct"),
        Prefix("regexp", "Regular Expression"),
        Prefix("string", "Text"),
        Prefix("date", "Date"),
        Prefix("time", "Time"),
        Prefix("enum_", "Enum"),
        Prefix("union", "Union"),
        Prefix("json", "JSON"),
        Prefix("to_", "Conversion"),
        Prefix("from_", "Conversion"),
        Prefix("is", "Predicate"),
        Prefix("bit", "Bitwise"),
        TypeRule(FuncTypes.SCALAR, "Scalar"),
        TypeRule(FuncTypes.AGGREGATE, "Aggregate"),
        TypeRule(FuncTypes.MACRO, "Macro"),
    )
)
"""Rules to categorize functions by name prefix or function type."""
