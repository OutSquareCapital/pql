from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import pyochain as pc

if TYPE_CHECKING:
    from ._frame import JoinKeys
TimeUnit = Literal["s", "ms", "us", "ns"]
JoinStrategy = Literal["inner", "left", "full", "cross", "semi", "anti"]
AsofJoinStrategy = Literal["backward", "forward", "nearest"]
UniqueKeepStrategy = Literal["any", "none", "first", "last"]

RoundMode = Literal["half_to_even", "half_away_from_zero"]
ClosedInterval = Literal["both", "left", "right", "none"]
RankMethod = Literal["average", "min", "max", "dense", "ordinal"]
FillNullStrategy = Literal["forward", "backward", "min", "max", "mean", "zero", "one"]
RollingInterpolationMethod = Literal["nearest", "higher", "lower", "midpoint", "linear"]


type JoinKeysRes[T: pc.Seq[str] | str] = pc.Result[JoinKeys[T], ValueError]
type OptIter[T] = pc.Option[T | Iterable[T]]
