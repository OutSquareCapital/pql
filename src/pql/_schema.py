from __future__ import annotations

from collections.abc import Callable
from typing import Self

import pyochain as pc
from pyochain.traits import PyoCollection

from . import sql
from ._datatypes import DataType

type ColumnResolver = Callable[[Schema], PyoCollection[str]]


class Schema(pc.Dict[str, DataType]):
    @classmethod
    def from_frame(cls, frame: sql.SqlFrame) -> Self:
        dtypes = frame.dtypes.iter().map(
            lambda d: DataType.__from_sql__(sql.parse_dtype(d))
        )
        return cls(frame.columns.iter().zip(dtypes, strict=True))
