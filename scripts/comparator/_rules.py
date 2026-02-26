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


type IgnoredParams = pc.Dict[Pql, pc.Dict[str, pc.Set[str]]]
IGNORED_PARAMS_BY_CLASS_AND_METHOD: IgnoredParams = pc.Dict(
    {
        Pql.LAZY_FRAME: pc.Dict.from_kwargs(
            sort=_args("maintain_order", _Arg.MULTITHREADED),
            join=_args(_Arg.ALLOW_PARALLEL, _Arg.FORCE_PARALLEL),
            join_asof=_args(_Arg.ALLOW_PARALLEL, _Arg.FORCE_PARALLEL),
            collect=_args("backend", "kwargs"),
        ),
        Pql.EXPR: pc.Dict.from_kwargs(sort_by=_args(_Arg.MULTITHREADED)),
        Pql.EXPR_LIST_NAME_SPACE: pc.Dict.from_kwargs(eval=_args(_Arg.PARALLEL)),
        Pql.EXPR_ARR_NAME_SPACE: pc.Dict.from_kwargs(eval=_args(_Arg.PARALLEL)),
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
