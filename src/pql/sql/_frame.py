from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar

import pyochain as pc

from ._code_gen import Relation
from ._creation import frame_init_into_duckdb, from_query
from .utils import TryIter, try_iter

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation

    from .typing import IntoRel, Orientation, PythonLiteral


class SqlFrame(Relation):
    _inner: DuckDBPyRelation
    __slots__: ClassVar[Iterable[str]] = ("_inner",)

    def __init__(self, data: IntoRel, orient: Orientation = "col") -> None:
        self._inner = frame_init_into_duckdb(data, orient=orient)

    def pivot(
        self,
        on: TryIter[str],
        using: TryIter[str] | None,
        group_by: TryIter[str] | None,
        in_values: Sequence[PythonLiteral] | None,
        order_by: TryIter[str] | None,
    ) -> SqlFrame:

        def _to_clause(vals: TryIter[str]) -> str:
            return try_iter(vals).join(", ")

        def _in_clause(vals: Sequence[PythonLiteral]) -> str:
            cols = (
                pc.Iter(vals)
                .map(lambda v: f"'{v}'" if isinstance(v, str) else str(v))
                .join(", ")
            )
            return f" IN ({cols})"

        def _clause(on: TryIter[str] | None, kword: str) -> str:
            return (
                pc.Option(on)
                .map(_to_clause)
                .map(lambda c: f" {kword} {c}")
                .unwrap_or("")
            )

        in_clause = pc.Option(in_values).map(_in_clause).unwrap_or("")
        query = f"""--sql
            PIVOT _rel
            ON {_to_clause(on)}{in_clause}{_clause(on=using, kword="USING")}{_clause(on=group_by, kword="GROUP BY")}{_clause(on=order_by, kword="ORDER BY")}"""
        return SqlFrame(from_query(query, _rel=self.inner()))
