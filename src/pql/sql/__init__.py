"""SQL expression functions and converters."""

from duckdb import table as from_table, table_function as from_table_function

from ._code_gen import Relation
from ._core import CoreHandler, try_flatten, try_iter
from ._datatypes import (
    ArrayType,
    DecimalType,
    DType,
    EnumType,
    ListType,
    MapType,
    RawTypes,
    ScalarType,
    SqlType,
    StructType,
    UnionType,
    parse_dtype,
)
from ._expr import SqlExpr
from ._funcs import (
    all,
    args_into_exprs,
    coalesce,
    col,
    element,
    fn_once,
    into_expr,
    lit,
    row_number,
    unnest,
    when,
)
from ._rel_conversions import (
    from_df,
    from_mapping,
    from_numpy,
    from_query,
    from_sequence,
)
from ._typing import IntoExpr, IntoExprColumn, IntoRel, NPArrayLike, SeqIntoVals

__all__ = [
    "ArrayType",
    "CoreHandler",
    "DType",
    "DecimalType",
    "EnumType",
    "IntoExpr",
    "IntoExprColumn",
    "IntoRel",
    "ListType",
    "MapType",
    "NPArrayLike",
    "RawTypes",
    "Relation",
    "ScalarType",
    "SeqIntoVals",
    "SqlExpr",
    "SqlType",
    "StructType",
    "UnionType",
    "all",
    "args_into_exprs",
    "coalesce",
    "col",
    "element",
    "fn_once",
    "from_df",
    "from_mapping",
    "from_numpy",
    "from_query",
    "from_sequence",
    "from_table",
    "from_table_function",
    "into_expr",
    "lit",
    "parse_dtype",
    "row_number",
    "try_flatten",
    "try_iter",
    "unnest",
    "when",
]
