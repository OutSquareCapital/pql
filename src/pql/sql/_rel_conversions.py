from collections.abc import Mapping, Sequence
from typing import Any, cast

import duckdb
import narwhals as nw
import pyochain as pc
from narwhals.typing import IntoFrame

from .typing import FrameLike, IntoDict, IntoRel, NPArrayLike, SeqIntoVals


def _unnest(k: str) -> duckdb.Expression:
    return duckdb.FunctionExpression("unnest", duckdb.ColumnExpression(k)).alias(k)


def frame_init_into_duckdb(data: IntoRel) -> duckdb.DuckDBPyRelation:  # noqa: PLR0911
    from ._core import DuckHandler

    match data:
        case duckdb.DuckDBPyRelation():
            return data
        case duckdb.Expression():
            return duckdb.values(data)
        case DuckHandler():
            return duckdb.values(data.inner())
        case Mapping():
            return from_dict(data)
        case NPArrayLike():
            return from_numpy(data)
        case FrameLike():
            return from_df(data)
        case Sequence():
            return from_records(data)


def from_query(query: str, **relations: IntoRel) -> duckdb.DuckDBPyRelation:
    """Create a relation from a SQL query."""

    def _as_namespace(
        rels: IntoDict[str, duckdb.DuckDBPyRelation],
    ) -> duckdb.DuckDBPyRelation:
        namespace = {"dk": duckdb, "qry": query, **dict(rels)}
        py_code = "relation = dk.from_query(qry)"
        exec(py_code, locals=namespace)  # noqa: S102
        return cast(duckdb.DuckDBPyRelation, namespace["relation"])

    return (
        pc.Dict(relations)
        .then(
            lambda d: (
                d.items().iter().map_star(lambda k, v: (k, frame_init_into_duckdb(v)))
            )
        )
        .map(_as_namespace)
        .unwrap_or_else(lambda: duckdb.from_query(query))
    )


def from_dicts(data: Sequence[Mapping[str, Any]]) -> duckdb.DuckDBPyRelation:
    return (
        pc.Iter(data[0].keys())
        .map(lambda key: (key, pc.Iter(data).map(lambda row: row[key]).collect(tuple)))
        .into(from_dict)
    )


def from_records(data: SeqIntoVals) -> duckdb.DuckDBPyRelation:
    match data[0]:
        case Mapping():
            vals = cast(Sequence[Mapping[str, Any]], data)
            return from_dicts(vals)
        case Sequence():
            vals = cast(Sequence[Sequence[Any]], data)
            return (
                pc.Iter(vals)
                .enumerate()
                .map_star(lambda k, v: (f"column_{k}", v))
                .into(from_dict)
            )
        case duckdb.Expression():
            vals = cast(Sequence[duckdb.Expression], data)
            return duckdb.values(tuple(vals))
        case _:
            col = "column_0"
            return duckdb.values(
                duckdb.ConstantExpression(tuple(data)).alias(col)
            ).select(_unnest(col))


def from_df(data: IntoFrame) -> duckdb.DuckDBPyRelation:
    match nw.from_native(data):
        case nw.DataFrame() as df:
            return df.lazy(backend="duckdb").to_native()
        case nw.LazyFrame() as lf:
            return duckdb.from_arrow(lf.collect())


def from_dict(data: IntoDict[str, Any]) -> duckdb.DuckDBPyRelation:
    data = pc.Dict(data)

    raw_vals = (
        data.items()
        .iter()
        .map_star(lambda k, v: duckdb.ConstantExpression(v).alias(k))
        .collect(tuple)
    )
    unnested = data.keys().iter().map(_unnest)
    return duckdb.values(raw_vals).select(*unnested)


def from_numpy(data: NPArrayLike[Any, Any]) -> duckdb.DuckDBPyRelation:
    _arr = data
    qry = """--sql
        SELECT *
        FROM _arr
        """
    return duckdb.from_query(qry)
