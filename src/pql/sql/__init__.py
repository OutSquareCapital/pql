"""SQL expression functions and converters."""

from duckdb import table as from_table, table_function as from_table_function

from . import typing
from ._code_gen import Relation
from ._core import CoreHandler
from ._datatypes import (
    ArrayType,
    DecimalType,
    DType,
    EnumType,
    ListType,
    MapType,
    ScalarType,
    SqlType,
    StructType,
    UnionType,
    parse_dtype,
)
from ._expr import (
    SqlExpr,
    SqlExprArrayNameSpace,
    SqlExprDateTimeNameSpace,
    SqlExprListNameSpace,
)
from ._funcs import (
    all,
    all_horizontal,
    any_horizontal,
    coalesce,
    col,
    element,
    fn_once,
    into_expr,
    lit,
    max_horizontal,
    mean_horizontal,
    min_horizontal,
    row_number,
    sum_horizontal,
    unnest,
)
from ._rel_conversions import (
    from_df,
    from_mapping,
    from_numpy,
    from_query,
    from_sequence,
)
from ._when import ChainedThen, ChainedWhen, Then, When, when
from ._window import Kword

__all__ = [
    "ArrayType",
    "ChainedThen",
    "ChainedWhen",
    "CoreHandler",
    "DType",
    "DecimalType",
    "EnumType",
    "Kword",
    "ListType",
    "MapType",
    "Relation",
    "ScalarType",
    "SqlExpr",
    "SqlExprArrayNameSpace",
    "SqlExprDateTimeNameSpace",
    "SqlExprListNameSpace",
    "SqlType",
    "StructType",
    "Then",
    "UnionType",
    "When",
    "all",
    "all_horizontal",
    "any_horizontal",
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
    "max_horizontal",
    "mean_horizontal",
    "min_horizontal",
    "parse_dtype",
    "row_number",
    "sum_horizontal",
    "typing",
    "unnest",
    "when",
]
