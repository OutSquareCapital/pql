import builtins
import keyword
from dataclasses import dataclass
from enum import StrEnum, auto

import pyochain as pc

from ._schemas import Categories


class PyTypes(StrEnum):
    """Python type names for DuckDB type mapping."""

    SELF = "Self"
    STR = auto()
    BOOL = auto()
    INT = auto()
    DATE = auto()
    FLOAT = auto()
    DECIMAL = "Decimal"
    BYTES = auto()
    BYTEARRAY = auto()
    MEMORYVIEW = auto()
    TIME = auto()
    DATETIME = auto()
    TIMEDELTA = auto()
    LIST = auto()
    DICT = "dict[object, object]"
    EXPR = "SqlExpr"
    NONE = "None"


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
    TIME = auto()
    TIMESTAMP = auto()
    TIMESTAMPTZ = auto()
    BITSTRING = auto()
    NULL = auto()


CONVERSION_MAP: pc.Dict[DuckDbTypes, pc.Seq[PyTypes]] = pc.Dict(
    {
        DuckDbTypes.VARCHAR: pc.Seq((PyTypes.STR,)),
        DuckDbTypes.INTEGER: pc.Seq((PyTypes.INT,)),
        DuckDbTypes.BIGINT: pc.Seq((PyTypes.INT,)),
        DuckDbTypes.SMALLINT: pc.Seq((PyTypes.INT,)),
        DuckDbTypes.TINYINT: pc.Seq((PyTypes.INT,)),
        DuckDbTypes.HUGEINT: pc.Seq((PyTypes.INT,)),
        DuckDbTypes.UINTEGER: pc.Seq((PyTypes.INT,)),
        DuckDbTypes.UBIGINT: pc.Seq((PyTypes.INT,)),
        DuckDbTypes.USMALLINT: pc.Seq((PyTypes.INT,)),
        DuckDbTypes.UTINYINT: pc.Seq((PyTypes.INT,)),
        DuckDbTypes.UHUGEINT: pc.Seq((PyTypes.INT,)),
        DuckDbTypes.DOUBLE: pc.Seq((PyTypes.FLOAT,)),
        DuckDbTypes.FLOAT: pc.Seq((PyTypes.FLOAT,)),
        DuckDbTypes.REAL: pc.Seq((PyTypes.FLOAT,)),
        DuckDbTypes.DECIMAL: pc.Seq((PyTypes.DECIMAL,)),
        DuckDbTypes.BOOLEAN: pc.Seq((PyTypes.BOOL,)),
        DuckDbTypes.DATE: pc.Seq((PyTypes.DATE,)),
        DuckDbTypes.TIME: pc.Seq((PyTypes.TIME,)),
        DuckDbTypes.TIMESTAMP: pc.Seq((PyTypes.DATETIME,)),
        DuckDbTypes.INTERVAL: pc.Seq((PyTypes.TIMEDELTA,)),
        DuckDbTypes.BLOB: pc.Seq(
            (PyTypes.BYTES, PyTypes.BYTEARRAY, PyTypes.MEMORYVIEW)
        ),
        DuckDbTypes.BIT: pc.Seq((PyTypes.BYTES, PyTypes.BYTEARRAY, PyTypes.MEMORYVIEW)),
        DuckDbTypes.UUID: pc.Seq((PyTypes.STR,)),
        DuckDbTypes.JSON: pc.Seq((PyTypes.STR,)),
        DuckDbTypes.ANY: pc.Seq((PyTypes.SELF,)),
        DuckDbTypes.LIST: pc.Seq((PyTypes.LIST,)),
        DuckDbTypes.MAP: pc.Seq((PyTypes.SELF,)),
        DuckDbTypes.STRUCT: pc.Seq((PyTypes.DICT,)),
        DuckDbTypes.ARRAY: pc.Seq((PyTypes.LIST,)),
        DuckDbTypes.UNION: pc.Seq((PyTypes.SELF,)),
        DuckDbTypes.TIMESTAMPTZ: pc.Seq((PyTypes.DATETIME,)),
        DuckDbTypes.BITSTRING: pc.Seq(
            (PyTypes.BYTES, PyTypes.BYTEARRAY, PyTypes.MEMORYVIEW)
        ),
        DuckDbTypes.NULL: pc.Seq((PyTypes.NONE,)),
    }
)
"""DuckDB type -> Python type hint mapping (exhaustive)."""

KWORDS = pc.Set(keyword.kwlist).union(
    pc.Iter(PyTypes)
    .filter(lambda t: t != PyTypes.SELF)
    .map(lambda t: t.value)
    .collect(pc.Set)
)
"""Python reserved keywords that need renaming when generating function names."""
SHADOWERS = KWORDS.union(pc.Set(dir(builtins))).union(pc.Set("l"))
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
class NamespaceSpec:
    name: str
    doc: str
    prefixes: pc.Seq[str]
    categories: pc.Seq[Categories]


NAMESPACE_SPECS = pc.Seq(
    (
        NamespaceSpec(
            name="ListFns",
            doc="Mixin providing auto-generated DuckDB list functions as methods.",
            prefixes=pc.Seq(("list_",)),
            categories=pc.Seq((Categories.LIST,)),
        ),
        NamespaceSpec(
            name="StructFns",
            doc="Mixin providing auto-generated DuckDB struct functions as methods.",
            prefixes=pc.Seq(("struct_",)),
            categories=pc.Seq((Categories.STRUCT,)),
        ),
        NamespaceSpec(
            name="StringFns",
            doc="Mixin providing auto-generated DuckDB string functions as methods.",
            prefixes=pc.Seq(("string_", "regexp_", "str")),
            categories=pc.Seq(
                (Categories.STRING, Categories.REGEX, Categories.TEXT_SIMILARITY)
            ),
        ),
        NamespaceSpec(
            name="DateTimeFns",
            doc="Mixin providing auto-generated DuckDB datetime functions as methods.",
            prefixes=pc.Seq(("date", "day")),
            categories=pc.Seq((Categories.TIMESTAMP,)),
        ),
        NamespaceSpec(
            name="ArrayFns",
            doc="Mixin providing auto-generated DuckDB array functions as methods.",
            prefixes=pc.Seq(("array_",)),
            categories=pc.Seq((Categories.ARRAY,)),
        ),
        NamespaceSpec(
            name="JsonFns",
            doc="Mixin providing auto-generated DuckDB JSON functions as methods.",
            prefixes=pc.Seq(("json_",)),
            categories=pc.Seq(()),
        ),
    )
)
"""Namespace metadata and function prefixes."""
