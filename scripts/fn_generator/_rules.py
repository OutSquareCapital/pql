import builtins
import keyword
from dataclasses import dataclass
from enum import StrEnum

import pyochain as pc

from .._utils import Builtins, DateTime, Decimal, Pql, Typing
from ._schemas import Categories, DuckDbTypes


def _py_types(*args: StrEnum) -> str:
    return pc.Iter(args).map(lambda pt: pt.value).join(" | ")


CONVERSION_MAP: pc.Dict[str, str] = pc.Dict(
    {
        DuckDbTypes.VARCHAR: _py_types(Builtins.STR),
        DuckDbTypes.INTEGER: _py_types(Builtins.INT),
        DuckDbTypes.BIGINT: _py_types(Builtins.INT),
        DuckDbTypes.SMALLINT: _py_types(Builtins.INT),
        DuckDbTypes.TINYINT: _py_types(Builtins.INT),
        DuckDbTypes.HUGEINT: _py_types(Builtins.INT),
        DuckDbTypes.UINTEGER: _py_types(Builtins.INT),
        DuckDbTypes.UBIGINT: _py_types(Builtins.INT),
        DuckDbTypes.USMALLINT: _py_types(Builtins.INT),
        DuckDbTypes.UTINYINT: _py_types(Builtins.INT),
        DuckDbTypes.UHUGEINT: _py_types(Builtins.INT),
        DuckDbTypes.DOUBLE: _py_types(Builtins.FLOAT),
        DuckDbTypes.FLOAT: _py_types(Builtins.FLOAT),
        DuckDbTypes.DECIMAL: _py_types(Decimal.DECIMAL),
        DuckDbTypes.BOOLEAN: _py_types(Builtins.BOOL),
        DuckDbTypes.DATE: _py_types(DateTime.DATE),
        DuckDbTypes.TIME: _py_types(DateTime.TIME),
        DuckDbTypes.TIMESTAMP: _py_types(DateTime.DATETIME),
        DuckDbTypes.INTERVAL: _py_types(DateTime.TIMEDELTA),
        DuckDbTypes.BLOB: _py_types(
            Builtins.BYTES, Builtins.BYTEARRAY, Builtins.MEMORYVIEW
        ),
        DuckDbTypes.BIT: _py_types(
            Builtins.BYTES, Builtins.BYTEARRAY, Builtins.MEMORYVIEW
        ),
        DuckDbTypes.UUID: _py_types(Builtins.STR),
        DuckDbTypes.JSON: _py_types(Builtins.STR),
        DuckDbTypes.ANY: _py_types(Typing.SELF),
        DuckDbTypes.LIST: _py_types(Builtins.LIST),
        DuckDbTypes.MAP: _py_types(Typing.SELF),
        DuckDbTypes.STRUCT: _py_types(Builtins.DICT),
        DuckDbTypes.ARRAY: _py_types(Builtins.LIST),
        DuckDbTypes.UNION: _py_types(Typing.SELF),
        DuckDbTypes.NULL: _py_types(Builtins.NONE),
    }
)


def _duckdb_type_to_py(enum_type: DuckDbTypes) -> str:
    def _base_py_for_value(value: str) -> str:
        mapped = CONVERSION_MAP.get_item(value).unwrap_or(Typing.SELF.value)
        return "" if mapped == Typing.SELF.value else mapped

    match enum_type.value:
        case inner if inner.endswith("[]") and enum_type not in (
            DuckDbTypes.ANY_ARRAY,
            DuckDbTypes.GENERIC_ARRAY,
            DuckDbTypes.V_ARRAY,
        ):
            return f"{Builtins.LIST}[{_base_py_for_value(inner.removesuffix('[]'))}]"
        case arr if "[" in arr:
            return _base_py_for_value(arr.partition("[")[0])
        case _ as value:
            return _base_py_for_value(value)


CONVERTER = pc.Iter(DuckDbTypes).map(lambda t: (t, _duckdb_type_to_py(t))).collect(dict)
"""DuckDB type -> Python type hint mapping."""

SHADOWERS = (
    Pql.into_iter()
    .chain(Typing)
    .chain(Builtins)
    .map(lambda s: s.value)
    .chain(dir(builtins))
    .insert("l")
    .collect(pc.Set)
    .union(pc.Set(keyword.kwlist))
)
"""Names that should be renamed to avoid shadowing."""
SPECIAL_CASES = pc.Set(
    {
        # "raw" operators
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
        "…",
        # Aliased operators
        "mod",
        "pow",
        "power",
        "add",
        "subtract",
        "multiply",
        "divide",
        # Conflicting names
        "alias",  # conflicts with duckdb alias method
        "list",  # conflict with namespace, renamed to implode
        "json",  # conflict with namespace, renamed to parse
        "map",  # conflict with namespace, renamed to to_map
        # Misc
        "log",  # Need to swap argument order to take self.inner() as value and not as base
        "strftime",  # Need custom "str" prefix rule, but this rule will also take "struct" funcs in string namespace, so better to just special case it
        "strptime",  # Same as strftime
        # Generic functions that cause too much conflicts with other names
        "greatest",  # Has 5 categories, same behavior across thoses, no namespace needed
        "least",  # Has 5 categories, same behavior across thoses, no namespace needed
        "concat",  # too much conflict with list_concat, array_concat, etc..
    }
)
"""Function to exclude by name, either because they require special handling or because they conflict with existing names."""
PREFIXES = pc.Set(
    (
        "__",  # Internal functions
        "current_",  # Utility fns
        "has_",  # Utility fns
        "pg_",  # Postgres fns
        "icu_",  # timestamp extension
    )
)
"""Functions to exclude by prefixes."""


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
            name="RegexFns",
            doc="Mixin providing auto-generated DuckDB regex functions as methods.",
            prefixes=pc.Seq(("regexp_",)),
            categories=pc.Seq((Categories.REGEX,)),
            strip_prefixes=pc.Seq(("regexp_", "str_", "string_")),
        ),
        NamespaceSpec(
            name="StringFns",
            doc="Mixin providing auto-generated DuckDB string functions as methods.",
            prefixes=pc.Seq(("string_", "str_")),
            categories=pc.Seq((Categories.STRING, Categories.TEXT_SIMILARITY)),
            strip_prefixes=pc.Seq(("string_", "str_")),
        ),
        NamespaceSpec(
            name="DateTimeFns",
            doc="Mixin providing auto-generated DuckDB datetime functions as methods.",
            prefixes=pc.Seq(("date", "day")),
            categories=pc.Seq((Categories.TIMESTAMP, Categories.DATE)),
            strip_prefixes=pc.Seq(("date_",)),
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
        NamespaceSpec(
            name="MapFns",
            doc="Mixin providing auto-generated DuckDB map functions as methods.",
            prefixes=pc.Seq(("map_",)),
            categories=pc.Seq(()),
            strip_prefixes=pc.Seq(("map_",)),
        ),
        NamespaceSpec(
            name="EnumFns",
            doc="Mixin providing auto-generated DuckDB enum functions as methods.",
            prefixes=pc.Seq(("enum_",)),
            categories=pc.Seq(()),
            strip_prefixes=pc.Seq(("enum_",)),
        ),
    )
)
"""Namespace metadata and function prefixes."""
