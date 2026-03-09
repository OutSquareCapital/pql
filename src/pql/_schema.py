from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Self

from pyochain.traits import PyoCollection, PyoMutableMapping

from . import sql
from ._datatypes import DataType

if TYPE_CHECKING:
    from .sql.typing import IntoDict

type ColumnResolver = Callable[[Schema], PyoCollection[str]]


class Schema(PyoMutableMapping[str, DataType]):
    _inner: dict[str, DataType]

    def __init__(self, data: IntoDict[str, DataType]) -> None:
        self._inner = dict(data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, key: str) -> DataType:
        return self._inner[key]

    def __setitem__(self, key: str, value: DataType) -> None:
        self._inner[key] = value

    def __delitem__(self, key: str) -> None:
        del self._inner[key]

    @classmethod
    def from_frame(cls, frame: sql.Relation) -> Self:
        dtypes = frame.dtypes.iter().map(
            lambda d: DataType.__from_sql__(sql.parse_dtype(d))
        )
        return cls(frame.columns.iter().zip(dtypes, strict=True))
