from __future__ import annotations

from typing import TYPE_CHECKING

from ._code_gen import Relation
from ._creation import frame_init_into_duckdb

if TYPE_CHECKING:
    from .typing import (
        IntoRel,
        Orientation,
    )


class SqlFrame(Relation):
    __slots__ = ()

    def __init__(self, data: IntoRel, orient: Orientation = "col") -> None:
        self._inner = frame_init_into_duckdb(data, orient=orient)
