from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from . import sql
from ._frame import LazyFrame

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame

    from .sql.typing import (
        IntoDict,
        IntoRel,
        NPArrayLike,
        Orientation,
        PythonLiteral,
        SeqIntoVals,
    )


def from_query(query: str, **relations: IntoRel) -> LazyFrame:
    return LazyFrame(sql.from_query(query, **relations))


def from_table(table: str) -> LazyFrame:
    return LazyFrame(sql.from_table(table))


def from_table_function(function: str) -> LazyFrame:
    return LazyFrame(sql.from_table_function(function))


def from_df(df: IntoFrame) -> LazyFrame:
    return LazyFrame(sql.from_df(df))


def from_numpy(arr: NPArrayLike[Any, Any], orient: Orientation = "col") -> LazyFrame:
    return LazyFrame(sql.from_numpy(arr, orient=orient))


def from_dict(mapping: IntoDict[str, PythonLiteral]) -> LazyFrame:
    return LazyFrame(sql.from_dict(mapping))


def from_dicts(mapping: Sequence[Mapping[str, PythonLiteral]]) -> LazyFrame:
    return LazyFrame(sql.from_dicts(mapping))


def from_records(data: SeqIntoVals) -> LazyFrame:
    return LazyFrame(sql.from_records(data))
