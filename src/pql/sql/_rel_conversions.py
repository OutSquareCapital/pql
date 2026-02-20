from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import duckdb
import narwhals as nw
import pyochain as pc
from narwhals.typing import IntoFrame

from ._typing import FrameLike, IntoRel, NPArrayLike, SeqIntoVals


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
            return duckdb.values(DuckHandler.into_duckdb(data))
        case Mapping():
            return from_mapping(data)
        case NPArrayLike():
            return from_numpy(data)
        case FrameLike():
            return from_df(data)
        case Sequence():
            return from_sequence(data)


def from_query(relations: pc.Dict[str, IntoRel], query: str) -> duckdb.DuckDBPyRelation:
    """Create a relation from a SQL query."""

    def _as_namespace(
        rels: Iterable[tuple[str, duckdb.DuckDBPyRelation]],
    ) -> duckdb.DuckDBPyRelation:
        namespace = {"dk": duckdb, "qry": query, **dict(rels)}
        py_code = "relation = dk.from_query(qry)"
        exec(py_code, locals=namespace)  # noqa: S102
        return cast(duckdb.DuckDBPyRelation, namespace["relation"])

    return (
        relations.then(
            lambda d: (
                d.items().iter().map_star(lambda k, v: (k, frame_init_into_duckdb(v)))
            )
        )
        .map(_as_namespace)
        .unwrap_or_else(lambda: duckdb.from_query(query))
    )


def from_sequence(data: SeqIntoVals) -> duckdb.DuckDBPyRelation:
    match data[0]:
        case Mapping():
            vals = pc.Seq(cast(Sequence[Mapping[str, Any]], data))
            return (
                pc.Iter(vals.first().keys())
                .map(
                    lambda key: (
                        key,
                        vals.iter().map(lambda row: row[key]).collect(tuple),
                    )
                )
                .into(from_mapping)
            )
        case Sequence():
            vals = cast(Sequence[Sequence[Any]], data)
            return (
                pc.Iter(vals)
                .enumerate()
                .map_star(lambda k, v: (f"column_{k}", v))
                .into(from_mapping)
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


def from_mapping(
    data: Mapping[str, Any] | Iterable[tuple[str, Any]],
) -> duckdb.DuckDBPyRelation:
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
