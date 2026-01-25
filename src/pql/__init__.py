"""PQL - Polars Query Language for SQL generation.

Provides a Polars-like API that generates SQL via sqlglot.
"""

# ==================== Module Exports ====================
from ._expr import Expr, ExprDateTimeNameSpace, ExprStringNameSpace, col, lit
from ._frame import GroupBy, LazyFrame

__all__ = [
    "Expr",
    "ExprDateTimeNameSpace",
    "ExprStringNameSpace",
    "GroupBy",
    "LazyFrame",
    "col",
    "lit",
]
