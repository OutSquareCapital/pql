from enum import StrEnum, auto

import pyochain as pc


class PyLit(StrEnum):
    INTO_DUCKDB = auto()
    SQLEXPR = "SqlExpr"
    DUCK_REL = "DuckDBPyRelation"
    DUCK_EXPR = "Expression"
    ITERABLE = "Iterable"
    NONE = "None"
    ANY = "Any"
    SELF_RET = "Self"
    LIST = auto()
    DICT = auto()
    OVERLOAD = auto()
    PROPERTY = auto()
    SELF = auto()
    STR = auto()


PYTYPING_REWRITES = pc.Dict.from_ref(
    {
        "pytyping.Any": PyLit.ANY,
        "pytyping.SupportsInt": "SupportsInt",
        "pytyping.List": "list",
        "pytyping.Literal": "Literal",
        "pytyping.Union": "Union",
        "pytorch": "torch",
        "pyarrow.lib.": "pa.",
        "pyarrow.": "pa.",
    }
)

TYPE_SUBS = pc.Dict({PyLit.DUCK_EXPR: PyLit.SQLEXPR, PyLit.DUCK_REL: PyLit.SELF_RET})


PARAM_TYPE_FIXES = pc.Dict.from_kwargs(
    aggregate=pc.Dict.from_kwargs(
        aggr_expr=f"{PyLit.DUCK_EXPR} | {PyLit.STR} | {PyLit.ITERABLE}[{PyLit.DUCK_EXPR} | {PyLit.STR}]"
    )
)

RETURN_TYPE_FIXES = pc.Dict.from_kwargs(dtypes="list[sqltypes.DuckDBPyType]")

KW_ONLY_FIXES = pc.Set({"write_csv", "write_parquet"})


def fix_rel_param(method_name: str, param_name: str, annotation: str) -> str:
    return (
        PARAM_TYPE_FIXES.get_item(method_name)
        .and_then(lambda params: params.get_item(param_name))
        .unwrap_or(annotation)
    )


def fix_rel_return(method_name: str, return_type: str) -> str:
    return RETURN_TYPE_FIXES.get_item(method_name).unwrap_or(return_type)


def fix_kw_only(method_name: str) -> bool:
    return KW_ONLY_FIXES.contains(method_name)
