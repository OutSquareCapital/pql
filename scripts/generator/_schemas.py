from dataclasses import asdict, dataclass, field
from enum import StrEnum, auto

import duckdb
import polars as pl
import pyochain as pc


class FuncTypes(StrEnum):
    """Function types in DuckDB."""

    PRAGMA = auto()
    TABLE = auto()
    TABLE_MACRO = auto()
    AGGREGATE = auto()
    MACRO = auto()
    SCALAR = auto()


FUNC_TYPES = pl.Enum(FuncTypes)


class Categories(StrEnum):
    """DuckDB function categories."""

    AGGREGATE = auto()
    ARRAY = auto()
    BITSTRING = auto()
    BLOB = auto()
    DATE = auto()
    LAMBDA = auto()
    LIST = auto()
    NUMERIC = auto()
    REGEX = auto()
    STRING = auto()
    STRUCT = auto()
    TEXT_SIMILARITY = auto()
    TIMESTAMP = auto()
    VARIANT = auto()


CATEGORY_TYPES = pl.Enum(Categories)


def schema(cls: type[object]) -> pl.Schema:
    """Simple decorator for creating Polars Schema from class attributes."""

    def _is_polars_dtype(_k: str, v: object) -> bool:
        return isinstance(v, (type, pl.DataType)) and (
            isinstance(v, pl.DataType) or issubclass(v, pl.DataType)
        )

    return (
        pc.Dict.from_object(cls)
        .items()
        .iter()
        .filter_star(_is_polars_dtype)
        .collect(pl.Schema)
    )


@schema
class TableSchema:
    """Schema for DuckDB functions table."""

    database_name = pl.String
    database_oid = pl.String
    schema_name = pl.String
    function_name = pl.String
    alias_of = pl.String()
    function_type = FUNC_TYPES
    description = pl.String()
    """Only present for scalar and aggregate functions."""
    comment = pl.String()
    tags = pl.List(pl.Struct({"key": pl.String, "value": pl.String}))
    return_type = pl.String()
    parameters = pl.List(pl.String)
    parameter_types = pl.List(pl.String)
    varargs = pl.String()
    macro_definition = pl.String()
    has_side_effects = pl.Boolean()
    internal = pl.Boolean()
    function_oid = pl.Int64()
    examples = pl.List(pl.String)
    stability = pl.String()
    categories = pl.List(CATEGORY_TYPES)


@dataclass(slots=True)
class ParamLens:
    by_fn: pl.Expr = field(default=pl.col("p_len_by_fn"))
    by_fn_cat: pl.Expr = field(default=pl.col("p_len_by_fn_cat"))
    by_fn_cat_desc: pl.Expr = field(default=pl.col("p_len_by_fn_cat_desc"))


@dataclass(slots=True)
class PyCols:
    name: pl.Expr = field(default=pl.col("py_name"))
    types: pl.Expr = field(default=pl.col("py_types"))


@dataclass(slots=True)
class Params:
    names: pl.Expr = field(default=pl.col("param_names"))
    idx: pl.Expr = field(default=pl.col("param_idx"))
    lens: ParamLens = field(default_factory=ParamLens)


@dataclass(slots=True)
class ParamLists:
    signatures: pl.Expr = field(default=pl.col("param_sig_list"))
    docs: pl.Expr = field(default=pl.col("param_doc_join"))
    names: pl.Expr = field(default=pl.col("param_names_join"))


@dataclass(slots=True)
class DuckCols:
    function_name: pl.Expr = field(default=pl.col("function_name"))
    function_type: pl.Expr = field(default=pl.col("function_type"))
    description: pl.Expr = field(default=pl.col("description"))
    categories: pl.Expr = field(default=pl.col("categories"))
    varargs: pl.Expr = field(default=pl.col("varargs"))
    alias_of: pl.Expr = field(default=pl.col("alias_of"))
    parameters: pl.Expr = field(default=pl.col("parameters"))
    parameter_types: pl.Expr = field(default=pl.col("parameter_types"))

    def query(self) -> duckdb.DuckDBPyRelation:

        cols = pc.Dict.from_ref(asdict(self)).keys().join(", ")
        qry = f"""--sql
            SELECT {cols}
            FROM duckdb_functions()
            """

        return duckdb.sql(qry)
