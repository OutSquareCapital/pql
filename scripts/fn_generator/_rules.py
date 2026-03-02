import builtins
import keyword
from dataclasses import dataclass

import pyochain as pc

from .._utils import Builtins, Pql, Typing
from ._dtypes import Categories, DuckDbTypes

CONVERTER = pc.Iter(DuckDbTypes).map(lambda t: (t, t.into_py())).collect(dict)
"""DuckDB type -> Python type hint mapping."""

SHADOWERS = (
    Pql.into_iter()
    .chain(Typing, Builtins)
    .map(lambda s: s.value)
    .chain(dir(builtins), keyword.kwlist)
    .insert("l")
    .collect(pc.Set)
)
"""Names that should be renamed to avoid shadowing."""

RENAME_RULES = pc.Dict.from_ref(
    {
        "list": "implode",
        "json": "json_parse",
        "map": "to_map",
        "kurtosis": "kurtosis_samp",
        "isnan": "is_nan",
        "isinf": "is_inf",
        "isfinite": "is_finite",
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
            prefixes=pc.Seq(("date", "day", "iso")),
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
