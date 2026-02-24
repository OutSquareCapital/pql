from enum import StrEnum, auto

import pyochain as pc

from .._utils import Pql


class RefBackend(StrEnum):
    NARWHALS = auto()
    POLARS = auto()


type IgnoredParams = pc.Dict[Pql, pc.Dict[str, pc.Set[str]]]
IGNORED_PARAMS_BY_CLASS_AND_METHOD: IgnoredParams = pc.Dict(
    {
        Pql.LAZY_FRAME: pc.Dict.from_kwargs(
            sort=pc.Set(("maintain_order", "multithreaded")),
            join=pc.Set(("allow_parallel", "force_parallel")),
            join_asof=pc.Set(("allow_parallel", "force_parallel")),
            collect=pc.Set(("backend", "kwargs")),
        ),
        Pql.EXPR_LIST_NAME_SPACE: pc.Dict.from_kwargs(eval=pc.Set(("parallel",))),
        Pql.EXPR_ARR_NAME_SPACE: pc.Dict.from_kwargs(eval=pc.Set(("parallel",))),
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
