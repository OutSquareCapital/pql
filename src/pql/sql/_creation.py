from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Self, cast

import duckdb
import narwhals as nw
import pyochain as pc
from sqlglot import exp

from ._datatypes import DType
from ._funcs import unnest
from .typing import FrameLike, NPArrayLike

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame

    from .typing import (
        AnyArray,
        IntoDict,
        IntoRel,
        Orientation,
        PythonLiteral,
        SeqIntoVals,
    )

COL0 = "column_0"


@dataclass(slots=True)
class Relation:
    rel: duckdb.DuckDBPyRelation
    schema: pc.Dict[str, DType]

    @classmethod
    def from_relation(cls, rel: duckdb.DuckDBPyRelation) -> Self:
        dtypes = pc.Iter(rel.dtypes).map(DType.parse)
        schma = pc.Iter(rel.columns).zip(dtypes, strict=True).collect(pc.Dict)
        return cls(rel, schma)

    @classmethod
    def from_table(cls, table: str) -> Self:
        return cls.from_relation(duckdb.table(table))

    @classmethod
    def from_table_function(cls, function: str) -> Self:
        return cls.from_relation(duckdb.table_function(function))

    @classmethod
    def from_query(cls, query: str, **relations: IntoRel) -> Self:
        """Create a relation from a SQL query."""

        def _as_namespace(
            rels: IntoDict[str, duckdb.DuckDBPyRelation],
        ) -> duckdb.DuckDBPyRelation:
            namespace = {"dk": duckdb, "qry": query, **dict(rels)}
            _PY_CODE(locals=namespace)
            return cast(duckdb.DuckDBPyRelation, namespace["relation"])

        return cls.from_relation(
            pc.Iter(relations.items())
            .map_star(lambda k, v: (k, into_relation(v).rel))
            .into(_as_namespace)
        )

    @classmethod
    def from_dicts(cls, data: Sequence[Mapping[str, PythonLiteral]]) -> Self:
        return (
            pc.Iter(data[0])
            .map(
                lambda key: (
                    key,
                    pc.Iter(data).map(lambda row: row[key]).collect(tuple),
                )
            )
            .into(cls.from_dict)
        )

    @classmethod
    def from_records(cls, data: SeqIntoVals) -> Self:
        from ._core import DuckHandler

        match data[0]:
            case Mapping():
                vals = cast(Sequence[Mapping[str, Any]], data)  # pyright: ignore[reportExplicitAny]
                return cls.from_dicts(vals)
            case Sequence():
                vals = cast(Sequence[Sequence[Any]], data)  # pyright: ignore[reportExplicitAny]
                return (
                    pc.Iter(vals)
                    .enumerate()
                    .map_star(lambda k, v: (_named(k), v))
                    .into(cls.from_dict)
                )
            case exp.Expr():
                vals = cast(Sequence[exp.Expr], data)

                return cls.from_relation(
                    duckdb.values(
                        pc.Iter(vals)
                        .map(lambda e: DuckHandler(e).into_duckdb())
                        .collect(tuple)
                    )
                )
            case _:
                return cls.from_relation(
                    duckdb.values(_to_expr(COL0, tuple(data))).select(_unnest(COL0))
                )

    @classmethod
    def from_df(cls, data: IntoFrame) -> Self:
        match nw.from_native(data):
            case nw.DataFrame() as df:
                return cls.from_relation(df.lazy(backend="duckdb").to_native())  # pyright: ignore[reportAny]
            case nw.LazyFrame() as lf:
                return cls.from_relation(duckdb.from_arrow(lf.collect()))

    @classmethod
    def from_dict(cls, data: IntoDict[str, Any]) -> Self:  # pyright: ignore[reportExplicitAny]
        data = pc.Dict(data)

        raw_vals = data.items().iter().map_star(_to_expr).collect(tuple)
        return cls.from_relation(
            duckdb.values(raw_vals).select(*data.iter().map(_unnest))
        )

    @classmethod
    def from_numpy(cls, data: AnyArray, orient: Orientation = "col") -> Self:

        match data.ndim:
            case 1:
                return cls.from_relation(
                    duckdb.values(_to_expr(COL0, data)).select(_unnest(COL0))
                )
            case _:
                arr = data.T if orient == "col" else data

                def _array_strategy() -> tuple[int, Callable[[int], AnyArray]]:
                    match (arr.ndim, orient):
                        case (2, _) | (_, "row"):
                            return 1, lambda j: arr[:, j]  # pyright: ignore[reportAny]
                        case _:
                            return 0, lambda j: arr[j]  # pyright: ignore[reportAny]

                def _named_array(names: pc.Seq[str]) -> duckdb.DuckDBPyRelation:
                    vals = (
                        names.iter()
                        .enumerate()
                        .map_star(lambda j, name: _to_expr(name, _arr_getter(j)))
                        .collect(tuple)
                    )

                    return duckdb.values(vals).select(*names.iter().map(_unnest))

                axis, _arr_getter = _array_strategy()
                names_nb: int = arr.shape[axis]  # pyright: ignore[reportAny]
                return cls.from_relation(
                    pc.Iter(range(names_nb)).map(_named).collect().into(_named_array)
                )


def _named(j: object) -> str:
    return f"column_{j}"


def _to_expr(k: str, v: PythonLiteral) -> duckdb.Expression:
    return duckdb.ConstantExpression(v).alias(k)


def _unnest(k: str) -> duckdb.Expression:
    return unnest(k).alias(k).into_duckdb()


def into_relation(  # noqa: PLR0911
    data: IntoRel, orient: Orientation = "col"
) -> Relation:
    from ._core import DuckHandler

    match data:
        case Relation():
            return data
        case duckdb.DuckDBPyRelation():
            return Relation.from_relation(data)
        case exp.Expr():
            return Relation.from_relation(
                duckdb.values(DuckHandler(data).into_duckdb())
            )
        case DuckHandler():
            return Relation.from_relation(duckdb.values(data.into_duckdb()))
        case Mapping():
            return Relation.from_dict(data)
        case NPArrayLike():
            return Relation.from_numpy(data, orient=orient)
        case FrameLike():
            return Relation.from_df(data)
        case Sequence():
            return Relation.from_records(data)


_QRY_ERR = "No relation provided"
_PY_CODE = partial(exec, "relation = dk.from_query(qry)")
