from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar, Self

import pyochain as pc

from ._code_gen import Relation
from ._creation import from_query, into_relation
from .utils import TryIter, try_iter

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation

    from .typing import IntoRel, Orientation, PythonLiteral


def _clause(on: TryIter[str], kword: str) -> str:
    return (
        pc.Option(on)
        .map(lambda o: try_iter(o).join(", "))
        .map(lambda c: f" {kword} {c}")
        .unwrap_or("")
    )


class SqlFrame(Relation):
    _inner: DuckDBPyRelation
    __slots__: ClassVar[Iterable[str]] = ("_inner",)

    def __init__(self, data: IntoRel, orient: Orientation = "col") -> None:
        self._inner = into_relation(data, orient=orient)

    def pivot(
        self,
        on: TryIter[str],
        using: TryIter[str],
        group_by: TryIter[str],
        in_values: Sequence[PythonLiteral] | None,
        order_by: TryIter[str],
    ) -> Self:

        def _in_clause(vals: Sequence[PythonLiteral]) -> str:
            cols = (
                pc.Iter(vals)
                .map(lambda v: f"'{v}'" if isinstance(v, str) else str(v))
                .join(", ")
            )
            return f" IN ({cols})"

        in_clause = pc.Option(in_values).map(_in_clause).unwrap_or("")

        qry = f"""--sql
        PIVOT _rel
        ON {try_iter(on).join(", ")}{in_clause}
        {_clause(on=using, kword="USING")}
        {_clause(on=group_by, kword="GROUP BY")}
        {_clause(on=order_by, kword="ORDER BY")}
        """

        return self.__class__(from_query(qry, _rel=self.inner()))

    def unpivot(
        self,
        on: TryIter[str],
        index: TryIter[str],
        variable_name: str,
        value_name: str,
        order_by: TryIter[str] = None,
    ) -> Self:
        """Unpivot from wide to long format."""
        index_cols = (
            pc.Option(index)
            .map(lambda value: try_iter(value).collect())
            .unwrap_or_else(pc.Seq[str].new)
        )
        on_cols = (
            pc.Option(on)
            .map(try_iter)
            .unwrap_or_else(
                lambda: self.columns.iter().filter(lambda name: name not in index_cols)
            )
            .join(", ")
        )
        slct_cols = index_cols.iter().chain((variable_name, value_name)).join(", ")

        qry = f"""--sql
        SELECT {slct_cols}
        FROM (UNPIVOT _rel ON {on_cols}
        INTO NAME {variable_name} VALUE {value_name})
        {_clause(on=order_by, kword="ORDER BY")}
        """

        return self.__class__(from_query(qry, _rel=self.inner()))
