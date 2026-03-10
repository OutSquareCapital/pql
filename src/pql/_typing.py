from typing import TYPE_CHECKING, Literal

import pyochain as pc

if TYPE_CHECKING:
    from ._frame import JoinKeys
type TimeUnit = Literal["ms", "us", "ns"]
type EpochTimeUnit = TimeUnit | Literal["s", "d"]
type JoinStrategy = Literal["inner", "left", "right", "outer", "semi", "anti", "cross"]
type AsofJoinStrategy = Literal["backward", "forward", "nearest"]
type UniqueKeepStrategy = Literal["any", "none", "first", "last"]
type TransferEncoding = Literal["hex", "base64"]
type ClosedInterval = Literal["both", "left", "right", "none"]
type RankMethod = Literal["average", "min", "max", "dense", "ordinal"]
type FillNullStrategy = Literal[
    "forward", "backward", "min", "max", "mean", "zero", "one"
]
type PivotAgg = Literal["min", "max", "first", "last", "sum", "mean", "median", "count"]
type JoinKeysRes[T: pc.Seq[str] | str] = pc.Result[JoinKeys[T], ValueError]
