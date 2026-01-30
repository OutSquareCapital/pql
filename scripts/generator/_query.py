import duckdb
import polars as pl

from ._models import CATEGORY_PATTERNS, FN_CATEGORY, KWORDS, SKIP_FUNCTIONS, FuncTypes


def sql_query() -> duckdb.DuckDBPyRelation:
    qry = """--sql
        SELECT
            function_name,
            function_type,
            return_type,
            parameters,
            parameter_types,
            varargs,
            description
        FROM duckdb_functions()
    """
    return duckdb.sql(qry)


def get_df() -> pl.DataFrame:
    fn_name: pl.Expr = pl.col("function_name")
    fn_type = pl.col("function_type")
    return (
        sql_query()
        .pl(lazy=True)
        .with_columns(
            fn_type.cast(FuncTypes).alias("function_type"),
            _normalize_list_expr("parameters"),
            _normalize_list_expr("parameter_types"),
            pl.col("return_type").fill_null("ANY").alias("return_type"),
            fn_name.pipe(_python_name_expr),
            fn_name.pipe(_category_expr),
            pl.col("parameters").list.len().alias("param_len"),
            pl.col("parameter_types").list.join(",").alias("_parameter_types_key"),
        )
        .filter(
            fn_type.is_in((FuncTypes.SCALAR, FuncTypes.AGGREGATE, FuncTypes.MACRO))
            .and_(fn_name.str.contains("##", literal=True).not_())
            .and_(fn_name.str.starts_with("_").not_())
            .and_(fn_name.str.contains(r"^[A-Za-z_][A-Za-z0-9_]*$"))
            .and_(fn_name.is_in(SKIP_FUNCTIONS).not_())
        )
        .with_columns(pl.col("param_len").min().over(fn_name).alias("min_param_len"))
        .sort(
            by=[fn_name, "param_len", "_parameter_types_key"],
            descending=[False, True, False],
        )
        .unique(subset=fn_name, keep="first")
        .select(
            fn_name,
            fn_type,
            "return_type",
            "python_name",
            "category",
            "parameters",
            "parameter_types",
            "varargs",
            "description",
            "min_param_len",
        )
        .collect()
    )


def _normalize_list_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column).is_null())
        .then(pl.lit([]).cast(pl.List(pl.String)))
        .otherwise(pl.col(column))
        .alias(column)
    )


def _python_name_expr(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr.is_in(KWORDS))
        .then(pl.concat_str([expr, pl.lit("_func")]))
        .otherwise(expr)
        .str.replace_all(r"([a-z0-9])([A-Z])", r"$1_$2")
        .str.to_lowercase()
        .alias("python_name")
    )


def _category_expr(expr: pl.Expr) -> pl.Expr:
    prefix_exprs = CATEGORY_PATTERNS.iter().map_star(
        lambda prefix, label: pl.when(expr.str.starts_with(prefix)).then(pl.lit(label))
    )
    type_exprs = (
        FN_CATEGORY.items()
        .iter()
        .map_star(
            lambda ftype, label: pl.when(pl.col("function_type").eq(ftype)).then(
                pl.lit(label)
            )
        )
    )
    return pl.coalesce((*prefix_exprs, *type_exprs, pl.lit("Other Functions"))).alias(
        "category"
    )
