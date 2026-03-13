from __future__ import annotations

from typing import Self

import pyochain as pc

from . import sql
from ._datatypes import DataType


class Schema(pc.Dict[str, DataType]):
    @classmethod
    def from_frame(cls, frame: sql.SqlFrame) -> Self:
        dtypes = frame.dtypes.iter().map(
            lambda d: DataType.__from_sql__(sql.parse_dtype(d))
        )
        return cls(frame.columns.iter().zip(dtypes, strict=True))
