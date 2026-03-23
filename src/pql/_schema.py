from __future__ import annotations

from dataclasses import dataclass

import pyochain as pc

from ._datatypes import DataType


@dataclass(slots=True, init=False)
class Schema(pc.Dict[str, DataType]):
    pass
