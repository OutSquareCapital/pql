import pprint
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
    function_type = pl.Enum(FuncTypes)
    description = pl.String()
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
    categories = pl.List(pl.String)


def main() -> None:
    qry = """--sql
    SELECT
        function_name,
        alias_of,
        function_type,
        description,
        return_type,
        parameters,
        parameter_types,
        varargs,
        macro_definition,
        examples,
        categories
    FROM duckdb_functions()
    """
    return (
        duckdb.sql(qry)
        .pl(lazy=True)
        .unique("function_name")
        .sort("function_name")
        .filter(
            pl.col("alias_of")
            .pipe(_has_val)
            .or_(pl.col("function_name").pipe(_has_val))
        )
        .collect()
        .pipe(lambda x: pprint.pprint(x.schema.keys()))
    )


def _has_val(expr: pl.Expr):
    return expr.str.contains("reduce", literal=True)


if __name__ == "__main__":
    main()
