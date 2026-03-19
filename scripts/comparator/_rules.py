from enum import StrEnum, auto

import pyochain as pc

from .._utils import Pql


def _args(*args: str) -> pc.Set[str]:
    return pc.Set(args)


class _Arg(StrEnum):
    ALLOW_PARALLEL = auto()
    PARALLEL = auto()
    FORCE_PARALLEL = auto()
    MULTITHREADED = auto()
    MORE_EXPRS = auto()
    MORE_PREDICATES = auto()
    INTERPOLATION = auto()


type IgnoredParams = pc.Dict[Pql, pc.Dict[str, pc.Set[str]]]
type IgnoredMembers = pc.Dict[Pql, pc.Set[str]]


IGNORED_MEMBERS: IgnoredMembers = pc.Dict(
    {
        Pql.MODULE_FUNCTIONS: _args(
            "show_versions",
            "col",
            "Utf8",
            "StringCache",
            "Series",
            "CredentialProvider",
            "CredentialProviderAWS",
            "CredentialProviderAzure",
            "CredentialProviderFunction",
            "CredentialProviderFunctionReturn",
            "CredentialProviderGCP",
            "DataFrame",
            "enable_string_cache",
            "unregister_extension_type",
            "using_string_cache",
            "get_extension_type",
            "generate_temporary_column_name",
            "QueryOptFlags",
            "SQLContext",
            "PartitionBy",
            "GPUEngine",
            "Implementation",
            "FileProviderArgs",
            "Config",
            "Categorical",
            "is_ordered_categorical",
            "disable_string_cache",
            "Categories",
        ),
        Pql.LAZY_FRAME: _args("cache", "clear", "with_columns_seq", "select_seq"),
        Pql.SELECTORS: _args("Selector", "categorical"),
    }
)

IGNORED_PARAMS: IgnoredParams = pc.Dict(
    {
        Pql.LAZY_FRAME: pc.Dict.from_kwargs(
            sort=_args("maintain_order", _Arg.MULTITHREADED),
            join=_args(_Arg.ALLOW_PARALLEL, _Arg.FORCE_PARALLEL),
            join_asof=_args(_Arg.ALLOW_PARALLEL, _Arg.FORCE_PARALLEL),
            collect=_args("backend", "kwargs"),
            quantile=_args(_Arg.INTERPOLATION),
            select=_args(_Arg.MORE_EXPRS),
            filter=_args(_Arg.MORE_PREDICATES),
            with_columns=_args(_Arg.MORE_EXPRS),
        ),
        Pql.EXPR: pc.Dict.from_kwargs(
            sort_by=_args(_Arg.MULTITHREADED), quantile=_args(_Arg.INTERPOLATION)
        ),
        Pql.EXPR_STRUCT_NAME_SPACE: pc.Dict.from_kwargs(
            with_fields=_args(_Arg.MORE_EXPRS)
        ),
        Pql.EXPR_LIST_NAME_SPACE: pc.Dict.from_kwargs(eval=_args(_Arg.PARALLEL)),
        Pql.EXPR_ARR_NAME_SPACE: pc.Dict.from_kwargs(eval=_args(_Arg.PARALLEL)),
        Pql.LAZY_GROUP_BY: pc.Dict.from_kwargs(
            agg=_args("more_aggs"),
            quantile=_args(_Arg.INTERPOLATION),
        ),
        Pql.MODULE_FUNCTIONS: pc.Dict.from_kwargs(
            all_horizontal=_args(_Arg.MORE_EXPRS),
            any_horizontal=_args(_Arg.MORE_EXPRS),
            max_horizontal=_args(_Arg.MORE_EXPRS),
            mean_horizontal=_args(_Arg.MORE_EXPRS),
            min_horizontal=_args(_Arg.MORE_EXPRS),
            sum_horizontal=_args(_Arg.MORE_EXPRS),
            when=_args(_Arg.MORE_PREDICATES),
        ),
    }
)


class Status(StrEnum):
    """Status of a method comparison."""

    MISSING = auto()
    SIGNATURE_MISMATCH = auto()
    MATCH = auto()
    EXTRA = auto()
