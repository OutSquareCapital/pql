from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import pyochain as pc

if TYPE_CHECKING:
    from ._frame import JoinKeys
type TimeUnit = Literal["s", "ms", "us", "ns"]
type JoinStrategy = Literal["inner", "left", "right", "outer", "semi", "anti", "cross"]
type AsofJoinStrategy = Literal["backward", "forward", "nearest"]
type UniqueKeepStrategy = Literal["any", "none", "first", "last"]
type TransferEncoding = Literal["hex", "base64"]
type RoundMode = Literal["half_to_even", "half_away_from_zero"]
type ClosedInterval = Literal["both", "left", "right", "none"]
type RankMethod = Literal["average", "min", "max", "dense", "ordinal"]
type FillNullStrategy = Literal[
    "forward", "backward", "min", "max", "mean", "zero", "one"
]
type RollingInterpolationMethod = Literal[
    "nearest", "higher", "lower", "midpoint", "linear"
]


type JoinKeysRes[T: pc.Seq[str] | str] = pc.Result[JoinKeys[T], ValueError]
type OptIter[T] = pc.Option[T | Iterable[T]]
