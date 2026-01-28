from typing import TYPE_CHECKING, Literal

import duckdb
import polars as pl
import pyochain as pc
from polars._typing import FrameInitTypes

if TYPE_CHECKING:
    from ._expr import Expr
RoundMode = Literal["half_to_even", "half_away_from_zero"]

type PyLiteral = str | int | float | bool | None
type FrameInit = (
    duckdb.DuckDBPyRelation | pl.DataFrame | pl.LazyFrame | None | FrameInitTypes
)
type IntoExpr = PyLiteral | Expr

type ByClause = pc.Seq[str] | pc.Seq[Expr] | pc.Seq[str | Expr]
type BoolClause = pc.Option[pc.Seq[bool]] | pc.Option[bool]
type SqlExpr = duckdb.Expression
