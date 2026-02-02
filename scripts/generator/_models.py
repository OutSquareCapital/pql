import builtins
import keyword
from enum import StrEnum, auto
from typing import Literal, NamedTuple

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


class CatRule(NamedTuple):
    rule_type: Literal["prefix", "type"]
    pattern: str
    label: str

    def into_expr(self, expr: pl.Expr) -> pl.Expr:
        match self.rule_type:
            case "prefix":
                return pl.when(expr.str.starts_with(self.pattern)).then(
                    pl.lit(self.label)
                )
            case "type":
                return pl.when(pl.col("function_type").eq(self.pattern)).then(
                    pl.lit(self.label)
                )


CATEGORY_RULES: pc.Seq[CatRule] = pc.Seq(
    (
        CatRule("prefix", "list", "List"),
        CatRule("prefix", "array", "Array"),
        CatRule("prefix", "map", "Map"),
        CatRule("prefix", "struct", "Struct"),
        CatRule("prefix", "regexp", "Regular Expression"),
        CatRule("prefix", "string", "Text"),
        CatRule("prefix", "date", "Date"),
        CatRule("prefix", "time", "Time"),
        CatRule("prefix", "enum_", "Enum"),
        CatRule("prefix", "union", "Union"),
        CatRule("prefix", "json", "JSON"),
        CatRule("prefix", "to_", "Conversion"),
        CatRule("prefix", "from_", "Conversion"),
        CatRule("prefix", "is", "Predicate"),
        CatRule("prefix", "bit", "Bitwise"),
        CatRule("type", FuncTypes.SCALAR, "Scalar"),
        CatRule("type", FuncTypes.AGGREGATE, "Aggregate"),
        CatRule("type", FuncTypes.MACRO, "Macro"),
    )
)
"""Rules to categorize functions by name prefix or function type."""
