from collections.abc import Iterable, Mapping
from typing import Any, cast

import duckdb
import narwhals as nw
import pyochain as pc

from pql.sql._typing import IntoFrame, IntoRel


def _from_df(data: IntoFrame) -> duckdb.DuckDBPyRelation:
    match nw.from_native(data):
        case nw.DataFrame() as df:
            return df.lazy(backend="duckdb").to_native()
        case nw.LazyFrame() as lf:
            return duckdb.from_arrow(lf.collect("pyarrow"))


def _from_mapping(data: Mapping[str, Any]) -> duckdb.DuckDBPyRelation:
    def _is_unnestable(value: object) -> bool:
        match value:
            case str() | bytes() | bytearray() | memoryview() | Mapping():
                return False
            case Iterable():
                return True
            case _:
                return False

    def _to_col(k: str, v: Any) -> duckdb.Expression:  # noqa: ANN401
        expr = duckdb.ConstantExpression(v)
        match _is_unnestable(v):
            case True:
                return duckdb.FunctionExpression("unnest", expr).alias(k)
            case False:
                return expr.alias(k)

    return pc.Dict(data).into(
        lambda mapped: duckdb.values(
            mapped.items()
            .iter()
            .map_star(lambda k, v: duckdb.ConstantExpression(v).alias(k))
            .collect(tuple)
        ).select(*mapped.items().iter().map_star(_to_col))
    )


def frame_init_into_duckdb(data: IntoRel) -> duckdb.DuckDBPyRelation:
    from ._core import DuckHandler

    match data:
        case duckdb.DuckDBPyRelation():
            return data
        case DuckHandler():
            return duckdb.values(DuckHandler.into_duckdb(data))
        case list() | tuple():
            return duckdb.values(data)
        case Mapping():
            return _from_mapping(data)
        case _:
            return _from_df(data)


def qry_into_duckdb(
    relations: pc.Dict[str, IntoRel], query: str
) -> duckdb.DuckDBPyRelation:
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
