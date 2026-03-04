from enum import StrEnum, auto

import pyochain as pc

from .._utils import Pql


class RefBackend(StrEnum):
    NARWHALS = auto()
    POLARS = auto()


def _args(*args: str) -> pc.Set[str]:
    return pc.Set(args)


class _Arg(StrEnum):
    ALLOW_PARALLEL = auto()
    PARALLEL = auto()
    FORCE_PARALLEL = auto()
    MULTITHREADED = auto()
    MORE_EXPRS = auto()
    INTERPOLATION = auto()


type IgnoredParams = pc.Dict[Pql, pc.Dict[str, pc.Set[str]]]
IGNORED_PARAMS: IgnoredParams = pc.Dict(
    {
        Pql.LAZY_FRAME: pc.Dict.from_kwargs(
            sort=_args("maintain_order", _Arg.MULTITHREADED),
            join=_args(_Arg.ALLOW_PARALLEL, _Arg.FORCE_PARALLEL),
            join_asof=_args(_Arg.ALLOW_PARALLEL, _Arg.FORCE_PARALLEL),
            collect=_args("backend", "kwargs"),
            quantile=_args("interpolation"),
            select=_args(_Arg.MORE_EXPRS),
            filter=_args("more_predicates"),
            with_columns=_args(_Arg.MORE_EXPRS),
        ),
        Pql.EXPR: pc.Dict.from_kwargs(
            sort_by=_args(_Arg.MULTITHREADED), quantile=_args(_Arg.INTERPOLATION)
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
        ),
    }
)


class Status(StrEnum):
    """Status of a method comparison."""

    MISSING = auto()
    SIGNATURE_MISMATCH = auto()
    MATCH = auto()
    EXTRA = auto()


class MismatchOn(StrEnum):
    """Source of a signature mismatch."""

    NW = auto()
    PL = auto()
    NULL = ""
