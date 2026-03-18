from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, cast

import duckdb
import narwhals as nw
import pyochain as pc
from narwhals.typing import IntoFrame

from .typing import (
    FrameLike,
    IntoDict,
    IntoRel,
    NPArrayLike,
    Orientation,
    PythonLiteral,
    SeqIntoVals,
)

COL0 = "column_0"


def _to_expr(k: str, v: PythonLiteral) -> duckdb.Expression:
    return duckdb.ConstantExpression(v).alias(k)


def _unnest(k: str) -> duckdb.Expression:
    return duckdb.FunctionExpression("unnest", duckdb.ColumnExpression(k)).alias(k)


def into_relation(  # noqa: PLR0911
    data: IntoRel, orient: Orientation = "col"
) -> duckdb.DuckDBPyRelation:
    from ._core import DuckHandler
    from ._frame import Frame

    match data:
        case duckdb.DuckDBPyRelation():
            return data
        case Frame():
            return data.inner()
        case duckdb.Expression():
            return duckdb.values(data)
        case DuckHandler():
            return duckdb.values(data.inner())
        case Mapping():
            return from_dict(data)
        case NPArrayLike():
            return from_numpy(data, orient=orient)
        case FrameLike():
            return from_df(data)
        case Sequence():
            return from_records(data)


_QRY_ERR = "No relation provided"
_PY_CODE = partial(exec, "relation = dk.from_query(qry)")


def from_query(query: str, **relations: IntoRel) -> duckdb.DuckDBPyRelation:
    """Create a relation from a SQL query."""

    def _as_namespace(
        rels: IntoDict[str, duckdb.DuckDBPyRelation],
    ) -> duckdb.DuckDBPyRelation:
        namespace = {"dk": duckdb, "qry": query, **dict(rels)}
        _PY_CODE(locals=namespace)
        return cast(duckdb.DuckDBPyRelation, namespace["relation"])

    return (
        pc.Iter(relations.items())
        .map_star(lambda k, v: (k, into_relation(v)))
        .into(_as_namespace)
    )


def from_dicts(data: Sequence[Mapping[str, PythonLiteral]]) -> duckdb.DuckDBPyRelation:
    return (
        pc.Iter(data[0])
        .map(lambda key: (key, pc.Iter(data).map(lambda row: row[key]).collect(tuple)))
        .into(from_dict)
    )


def from_records(data: SeqIntoVals) -> duckdb.DuckDBPyRelation:
    match data[0]:
        case Mapping():
            vals = cast(Sequence[Mapping[str, Any]], data)  # pyright: ignore[reportExplicitAny]
            return from_dicts(vals)
        case Sequence():
            vals = cast(Sequence[Sequence[Any]], data)  # pyright: ignore[reportExplicitAny]
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
            return duckdb.values(_to_expr(COL0, tuple(data))).select(_unnest(COL0))


def from_df(data: IntoFrame) -> duckdb.DuckDBPyRelation:
    match nw.from_native(data):
        case nw.DataFrame() as df:
            return df.lazy(backend="duckdb").to_native()  # pyright: ignore[reportAny]
        case nw.LazyFrame() as lf:
            return duckdb.from_arrow(lf.collect())


def from_dict(data: IntoDict[str, Any]) -> duckdb.DuckDBPyRelation:  # pyright: ignore[reportExplicitAny]
    data = pc.Dict(data)

    raw_vals = data.items().iter().map_star(_to_expr).collect(tuple)
    unnested = data.iter().map(_unnest)
    return duckdb.values(raw_vals).select(*unnested)


def from_numpy(
    data: NPArrayLike[Any, Any],  # pyright: ignore[reportExplicitAny]
    orient: Orientation = "col",
) -> duckdb.DuckDBPyRelation:

    match data.ndim:
        case 1:
            return duckdb.values(_to_expr(COL0, data)).select(_unnest(COL0))
        case _:
            arr = data.T if orient == "col" else data

            def _array_strategy(
                data: NPArrayLike[Any, Any],  # pyright: ignore[reportExplicitAny]
            ) -> tuple[int, Callable[[int], NPArrayLike[Any, Any]]]:  # pyright: ignore[reportExplicitAny]
                match (data.ndim, orient):
                    case (2, _) | (_, "row"):

                        def _arr_getter(j: int) -> NPArrayLike[Any, Any]:  # pyright: ignore[reportExplicitAny]
                            return arr[:, j]  # pyright: ignore[reportAny]

                        return 1, _arr_getter
                    case _:

                        def _arr_getter(j: int) -> NPArrayLike[Any, Any]:  # pyright: ignore[reportExplicitAny]
                            return arr[j]  # pyright: ignore[reportAny]

                        return 0, _arr_getter

            axis, _arr_getter = _array_strategy(arr)

            names = (
                pc.Iter(range(arr.shape[axis])).map(lambda j: f"column_{j}").collect()  # pyright: ignore[reportAny]
            )
            vals = (
                names.iter()
                .enumerate()
                .map_star(lambda j, name: _to_expr(name, _arr_getter(j)))
                .collect(tuple)
            )

            return duckdb.values(vals).select(*names.iter().map(_unnest))
