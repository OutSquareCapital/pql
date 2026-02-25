import builtins
import keyword
from dataclasses import dataclass

import pyochain as pc

from .._utils import Builtins, DateTime, Decimal, Pql, Typing
from ._schemas import Categories, DuckDbTypes

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
        DuckDbTypes.LIST: Builtins.LIST.value,
        DuckDbTypes.MAP: Typing.SELF.value,
        DuckDbTypes.STRUCT: Builtins.DICT.value,
        DuckDbTypes.ARRAY: Builtins.LIST.value,
        DuckDbTypes.UNION: Typing.SELF.value,
        DuckDbTypes.NULL: Builtins.NONE.value,
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
            return Builtins.LIST.of_type(_base_py_for_value(inner.removesuffix("[]")))
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

RENAME_RULES = pc.Dict.from_ref(
    {
        "list": "implode",
        "json": "json_parse",
        "map": "to_map",
        "kurtosis": "kurtosis_samp",
    }
)
"""Explicit SQL function name -> generated Python method name mapping."""


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
        # Already exist in duckdb Expression methods
        "mod",
        "pow",
        "power",
        "add",
        "subtract",
        "multiply",
        "divide",
        "alias",  # conflicts with duckdb alias method
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
