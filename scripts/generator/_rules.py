import builtins
import keyword
from dataclasses import dataclass
from enum import StrEnum, auto

import pyochain as pc

from ._schemas import Categories, DuckDbTypes


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


def _py_types(*args: PyTypes) -> str:
    return pc.Iter(args).map(lambda pt: pt.value).join(" | ")


CONVERSION_MAP: pc.Dict[str, str] = pc.Dict(
    {
        DuckDbTypes.VARCHAR: _py_types(PyTypes.STR),
        DuckDbTypes.INTEGER: _py_types(PyTypes.INT),
        DuckDbTypes.BIGINT: _py_types(PyTypes.INT),
        DuckDbTypes.SMALLINT: _py_types(PyTypes.INT),
        DuckDbTypes.TINYINT: _py_types(PyTypes.INT),
        DuckDbTypes.HUGEINT: _py_types(PyTypes.INT),
        DuckDbTypes.UINTEGER: _py_types(PyTypes.INT),
        DuckDbTypes.UBIGINT: _py_types(PyTypes.INT),
        DuckDbTypes.USMALLINT: _py_types(PyTypes.INT),
        DuckDbTypes.UTINYINT: _py_types(PyTypes.INT),
        DuckDbTypes.UHUGEINT: _py_types(PyTypes.INT),
        DuckDbTypes.DOUBLE: _py_types(PyTypes.FLOAT),
        DuckDbTypes.FLOAT: _py_types(PyTypes.FLOAT),
        DuckDbTypes.DECIMAL: _py_types(PyTypes.DECIMAL),
        DuckDbTypes.BOOLEAN: _py_types(PyTypes.BOOL),
        DuckDbTypes.DATE: _py_types(PyTypes.DATE),
        DuckDbTypes.TIME: _py_types(PyTypes.TIME),
        DuckDbTypes.TIMESTAMP: _py_types(PyTypes.DATETIME),
        DuckDbTypes.INTERVAL: _py_types(PyTypes.TIMEDELTA),
        DuckDbTypes.BLOB: _py_types(
            PyTypes.BYTES, PyTypes.BYTEARRAY, PyTypes.MEMORYVIEW
        ),
        DuckDbTypes.BIT: _py_types(
            PyTypes.BYTES, PyTypes.BYTEARRAY, PyTypes.MEMORYVIEW
        ),
        DuckDbTypes.UUID: _py_types(PyTypes.STR),
        DuckDbTypes.JSON: _py_types(PyTypes.STR),
        DuckDbTypes.ANY: _py_types(PyTypes.SELF),
        DuckDbTypes.LIST: _py_types(PyTypes.LIST),
        DuckDbTypes.MAP: _py_types(PyTypes.SELF),
        DuckDbTypes.STRUCT: _py_types(PyTypes.DICT),
        DuckDbTypes.ARRAY: _py_types(PyTypes.LIST),
        DuckDbTypes.UNION: _py_types(PyTypes.SELF),
        DuckDbTypes.NULL: _py_types(PyTypes.NONE),
    }
)


def _duckdb_type_to_py(enum_type: DuckDbTypes) -> str:
    def _base_py_for_value(value: str) -> str:
        mapped = CONVERSION_MAP.get_item(value).unwrap_or(PyTypes.SELF.value)
        return "" if mapped == PyTypes.SELF.value else mapped

    match enum_type.value:
        case inner if inner.endswith("[]") and enum_type not in (
            DuckDbTypes.ANY_ARRAY,
            DuckDbTypes.GENERIC_ARRAY,
            DuckDbTypes.V_ARRAY,
        ):
            return f"list[{_base_py_for_value(inner.removesuffix('[]'))}]"
        case arr if "[" in arr:
            return _base_py_for_value(arr.partition("[")[0])
        case _ as value:
            return _base_py_for_value(value)


CONVERTER = pc.Iter(DuckDbTypes).map(lambda t: (t, _duckdb_type_to_py(t))).collect(dict)
"""DuckDB type -> Python type hint mapping."""

SHADOWERS = pc.Set(keyword.kwlist).union(
    pc.Iter(PyTypes)
    .filter(lambda t: t != PyTypes.SELF)
    .map(lambda t: t.value)
    .collect(pc.Set)
    .union(pc.Set(dir(builtins)))
    .union(pc.Set("l"))
)
"""Names that should be renamed to avoid shadowing."""
SPECIAL_CASES = pc.Set(
    {
        "+",
        "-",
        "/",
        "*",
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
        "alias",  # conflicts with duckdb alias method
        "log",  # Need to swap argument order to take self._expr as value and not as base
        "list",  # conflict with namespace, renamed to implode
        "add",  # conflict with __add__ and add method. and is surely redundant
        "mod",  # conflict with __mod__ and mod method. and is surely redundant
        "pow",  # conflict with __pow__ and pow method. and is surely redundant
    }
)
"""Function to exclude."""
PREFIXES = pc.Set(
    (
        "__",
        "current_",  # Utility fns
        "has_",  # Utility fns
        "pg_",  # Postgres fns
        "icu_",  # timestamp extension
    )
)


@dataclass(slots=True)
class NamespaceSpec:
    name: str
    doc: str
    prefixes: pc.Seq[str]
    categories: pc.Seq[Categories]
    strip_prefixes: pc.Seq[str]


NAMESPACE_SPECS = pc.Seq(
    (
        NamespaceSpec(
            name="ListFns",
            doc="Mixin providing auto-generated DuckDB list functions as methods.",
            prefixes=pc.Seq(("list_",)),
            categories=pc.Seq((Categories.LIST,)),
            strip_prefixes=pc.Seq(("list_", "array_")),
        ),
        NamespaceSpec(
            name="StructFns",
            doc="Mixin providing auto-generated DuckDB struct functions as methods.",
            prefixes=pc.Seq(("struct_",)),
            categories=pc.Seq((Categories.STRUCT,)),
            strip_prefixes=pc.Seq(("struct_",)),
        ),
        NamespaceSpec(
            name="StringFns",
            doc="Mixin providing auto-generated DuckDB string functions as methods.",
            prefixes=pc.Seq(("string_", "regexp_", "str")),
            categories=pc.Seq(
                (Categories.STRING, Categories.REGEX, Categories.TEXT_SIMILARITY)
            ),
            strip_prefixes=pc.Seq(("string_", "regexp_", "str_")),
        ),
        NamespaceSpec(
            name="DateTimeFns",
            doc="Mixin providing auto-generated DuckDB datetime functions as methods.",
            prefixes=pc.Seq(("date", "day")),
            categories=pc.Seq((Categories.TIMESTAMP,)),
            strip_prefixes=pc.Seq(()),
        ),
        NamespaceSpec(
            name="ArrayFns",
            doc="Mixin providing auto-generated DuckDB array functions as methods.",
            prefixes=pc.Seq(("array_",)),
            categories=pc.Seq((Categories.ARRAY,)),
            strip_prefixes=pc.Seq(("array_",)),
        ),
        NamespaceSpec(
            name="JsonFns",
            doc="Mixin providing auto-generated DuckDB JSON functions as methods.",
            prefixes=pc.Seq(("json_",)),
            categories=pc.Seq(()),
            strip_prefixes=pc.Seq(("json_",)),
        ),
    )
)
"""Namespace metadata and function prefixes."""
