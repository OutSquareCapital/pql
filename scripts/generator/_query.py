from dataclasses import dataclass, field

import duckdb
import polars as pl

from ._models import (
    CATEGORY_RULES,
    CONVERSION_MAP,
    FUNC_TYPES,
    KWORDS,
    OPERATOR_MAP,
    SHADOWERS,
    DuckDbTypes,
    FuncTypes,
    PyTypes,
)

_EMPTY_STR = pl.lit("")


def sql_query() -> duckdb.DuckDBPyRelation:
    qry = """--sql
        SELECT *
        FROM duckdb_functions()
    """
    return duckdb.sql(qry)


@dataclass(slots=True)
class ParamLens:
    by_func: pl.Expr = field(default=pl.col("p_len_by_func"))
    by_func_and_cat: pl.Expr = field(default=pl.col("p_len_by_func_and_cat"))
    by_func_cat_and_desc: pl.Expr = field(default=pl.col("p_len_by_func_cat_and_desc"))


def get_df() -> pl.LazyFrame:
    _fn_name = pl.col("function_name")
    _params = pl.col("parameters")
    _py_name = pl.col("python_name")
    _py_types = pl.col("py_types")
    _param_types = pl.col("parameter_types")
    _param_names = pl.col("param_names")
    _category = pl.col("category")
    _description = pl.col("description")
    _varargs = pl.col("varargs")
    _duckdb_cats = pl.col("categories")
    _param_idx = pl.col("param_idx")
    _py_type_union = pl.col("py_types_union")
    p_lens = ParamLens()

    return (
        sql_query()
        .pl()
        .filter(
            pl.col("function_type")
            .cast(FUNC_TYPES)
            .is_in((FuncTypes.SCALAR, FuncTypes.AGGREGATE, FuncTypes.MACRO))
            .and_(_fn_name.str.starts_with("__").not_())
            .and_(_fn_name.is_in(OPERATOR_MAP).not_())
            .and_(
                pl.col("alias_of").pipe(
                    lambda expr: expr.is_null().or_(expr.is_in(OPERATOR_MAP))
                )
            )
        )
        .with_columns(
            _param_types.list.eval(
                pl.element().fill_null(DuckDbTypes.ANY.value.upper())
            ),
            *_params.list.len().pipe(
                lambda expr_len: (
                    expr_len.alias("p_len_by_func"),
                    expr_len.min().over(_fn_name).alias("p_len_by_func_and_cat"),
                    expr_len.min()
                    .over(_fn_name, _duckdb_cats, _description)
                    .alias("p_len_by_func_cat_and_desc"),
                )
            ),
            _varargs.pipe(_convert_duckdb_type_to_python)
            .pipe(_make_type_union)
            .alias("varargs_py_type"),
        )
        .with_columns(
            _duckdb_cats.pipe(_to_py_name, _fn_name, _description, p_lens, _params)
        )
        .with_row_index("sig_id")
        .explode("parameters", "parameter_types")
        .with_columns(
            pl.int_range(pl.len()).over("sig_id").alias("param_idx"),
            _param_types.pipe(_convert_duckdb_type_to_python).alias("py_types"),
            _params.pipe(_to_param_names, _py_name),
        )
        .group_by(_py_name, _param_idx, maintain_order=True)
        .agg(
            pl.all()
            .exclude("param_doc_join", "py_types", "parameter_types")
            .drop_nulls()
            .first(),
            _param_types.unique().str.join(" | ").alias("param_types_union"),
            _py_types.unique().str.join(" | ").alias("py_types_union"),
        )
        .sort(_py_name, _param_idx)
        .group_by(_py_name, maintain_order=True)
        .agg(
            pl.all().exclude("param_names", "param_doc_join").first(),
            _py_type_union.str.contains(r"\bbool\b")
            .filter(_param_names.is_not_null())
            .alias("param_is_bool"),
            *_param_names.filter(_param_names.is_not_null()).pipe(
                lambda expr: (
                    expr.alias("param_names_list"),
                    expr.str.join(", ").alias("param_names_join"),
                    *_param_idx.ge(p_lens.by_func_and_cat).pipe(
                        lambda cond: (
                            pl.concat_str(
                                expr,
                                pl.lit(": "),
                                _py_type_union.pipe(_make_type_union),
                                pl.when(cond)
                                .then(pl.lit(" | None"))
                                .otherwise(_EMPTY_STR),
                                pl.when(cond)
                                .then(pl.lit(" = None"))
                                .otherwise(_EMPTY_STR),
                            ).alias("param_sig_list"),
                            pl.concat_str(
                                pl.lit("        "),
                                expr,
                                pl.lit(" ("),
                                _py_type_union.pipe(_make_type_union),
                                pl.when(cond)
                                .then(pl.lit(" | None"))
                                .otherwise(_EMPTY_STR),
                                pl.lit("): `"),
                                pl.col("param_types_union"),
                                pl.lit("` expression"),
                            )
                            .str.join("\n")
                            .alias("param_doc_join"),
                        )
                    ),
                )
            ),
        )
        .select(
            pl.coalesce(
                CATEGORY_RULES.iter().map(lambda cat: cat.into_expr(_py_name))
            ).alias("category"),
            _py_name,
            pl.col("param_names_list").pipe(
                _to_func,
                _py_name,
                _description,
                _varargs.is_not_null(),
                _varargs,
                _fn_name,
                pl.col("param_is_bool"),
            ),
        )
        .sort(_category, _py_name)
        .lazy()
    )


def _make_type_union(py_type: pl.Expr) -> pl.Expr:
    return (
        pl.when(py_type.eq(pl.lit(PyTypes.EXPR.value)))
        .then(py_type)
        .otherwise(pl.concat_str(pl.lit(PyTypes.EXPR.value), pl.lit(" | "), py_type))
    )


def _convert_duckdb_type_to_python(param_type: pl.Expr) -> pl.Expr:
    return param_type.str.extract(r"^([A-Z]+)", 1).replace_strict(
        CONVERSION_MAP.items()
        .iter()
        .map_star(lambda k, v: (k.value.upper(), v.value))
        .collect(dict),
        default=PyTypes.EXPR.value,
        return_dtype=pl.String,
    )


def _to_param_names(_params: pl.Expr, _py_name: pl.Expr) -> pl.Expr:
    return (
        _params.str.strip_chars_start("'\"[")
        .str.strip_chars_end("'\"[]")
        .str.replace(r"\(.*$", _EMPTY_STR)
        .str.replace_all(r"\.\.\.", _EMPTY_STR)
        .pipe(
            lambda expr: pl.when(expr.is_in(SHADOWERS))
            .then(pl.concat_str(expr, pl.lit("_arg")))
            .otherwise(expr)
        )
        .pipe(
            lambda expr: pl.when(expr.str.contains(r"^[A-Za-z_][A-Za-z0-9_]*$").not_())
            .then(_EMPTY_STR)
            .otherwise(expr)
        )
        .pipe(
            lambda expr: (
                pl.when(expr.eq(_EMPTY_STR))
                .then(
                    pl.concat_str(
                        pl.lit("_arg"),
                        expr.cum_count().over(_py_name).cast(pl.String),
                    )
                )
                .otherwise(expr)
            )
        )
        .pipe(
            lambda expr: (
                pl.when(expr.cum_count().over(_py_name, expr).gt(1))
                .then(
                    pl.concat_str(
                        expr,
                        pl.lit("_"),
                        expr.cum_count().over(_py_name, expr).cast(pl.String),
                    )
                )
                .otherwise(expr)
            )
        )
        .alias("param_names")
    )


def _to_py_name(
    duckdb_cats: pl.Expr,
    fn_name: pl.Expr,
    description: pl.Expr,
    p_lens: ParamLens,
    params: pl.Expr,
) -> pl.Expr:
    return duckdb_cats.list.join("_").pipe(
        lambda cat_str: fn_name.pipe(
            lambda expr: (
                pl.when(expr.is_in(KWORDS))
                .then(pl.concat_str(expr, pl.lit("_func")))
                .otherwise(expr)
                .str.replace_all(r"([a-z0-9])([A-Z])", r"$1_$2")
                .str.to_lowercase()
            )
        )
        .pipe(
            lambda base_name: pl.when(
                cat_str.n_unique().over(fn_name).gt(1).and_(cat_str.ne(_EMPTY_STR))
            )
            .then(pl.concat_str(base_name, pl.lit("_"), cat_str))
            .otherwise(base_name)
        )
        .pipe(
            lambda base: pl.when(
                description.n_unique()
                .over(fn_name, duckdb_cats)
                .gt(1)
                .and_(p_lens.by_func_cat_and_desc.gt(p_lens.by_func_and_cat))
            )
            .then(
                pl.concat_str(
                    base,
                    pl.lit("_"),
                    pl.when(p_lens.by_func.eq(p_lens.by_func_cat_and_desc))
                    .then(
                        pl.when(p_lens.by_func_cat_and_desc.eq(p_lens.by_func_and_cat))
                        .then(params)
                        .otherwise(
                            params.list.slice(
                                p_lens.by_func_and_cat,
                                p_lens.by_func.sub(p_lens.by_func_and_cat),
                            )
                        )
                    )
                    .otherwise(pl.lit([], dtype=pl.List(pl.String)))
                    .list.join("_")
                    .str.to_lowercase()
                    .str.replace_all(r"[^A-Za-z0-9_]+", "_")
                    .str.replace_all(r"_+", "_")
                    .str.strip_chars("_")
                    .max()
                    .over(fn_name, duckdb_cats, description),
                )
            )
            .otherwise(base)
        )
        .alias("python_name")
    )


def _to_func(
    expr: pl.Expr,
    py_name: pl.Expr,
    description: pl.Expr,
    has_varargs: pl.Expr,
    varargs: pl.Expr,
    fn_name: pl.Expr,
    is_bool: pl.Expr,
) -> pl.Expr:
    def _signature(has_params: pl.Expr) -> pl.Expr:
        return pl.concat_str(
            pl.when(has_params.or_(has_varargs))
            .then(
                pl.concat_str(
                    pl.lit("def "),
                    py_name,
                    pl.lit("("),
                    is_bool.pipe(
                        lambda p_is_bool_list: (
                            pl.when(p_is_bool_list.list.any())
                            .then(p_is_bool_list.list.arg_max())
                            .otherwise(pl.lit(0))
                            .pipe(
                                lambda first_bool_idx: pl.when(
                                    p_is_bool_list.list.any().and_(has_varargs.not_())
                                )
                                .then(
                                    pl.col("param_sig_list").pipe(
                                        lambda p_sig_list: pl.concat_list(
                                            p_sig_list.list.slice(0, first_bool_idx),
                                            pl.lit(["*"]),
                                            p_sig_list.list.slice(
                                                first_bool_idx,
                                                p_sig_list.list.len(),
                                            ),
                                        )
                                    )
                                )
                                .otherwise(pl.col("param_sig_list"))
                                .list.join(", ")
                            )
                        )
                    ),
                    pl.when(has_varargs)
                    .then(
                        pl.concat_str(
                            pl.when(has_params)
                            .then(pl.lit(", *args: "))
                            .otherwise(pl.lit("*args: ")),
                            pl.col("varargs_py_type"),
                        )
                    )
                    .otherwise(_EMPTY_STR),
                    pl.lit(f") -> {PyTypes.EXPR}:"),
                )
            )
            .otherwise(
                pl.concat_str(pl.lit("def "), py_name, pl.lit(f"() -> {PyTypes.EXPR}:"))
            )
        )

    def _description() -> pl.Expr:
        return (
            pl.when(description.is_not_null())
            .then(
                description.str.strip_chars()
                .str.replace_all("\u2019", "'")
                .str.replace_all(r"\. ", ".\n\n    ")
                .str.strip_chars_end(".")
                .add(pl.lit("."))
            )
            .otherwise(pl.concat_str(pl.lit("SQL "), fn_name, pl.lit(" function.")))
        )

    def _args_section(has_params: pl.Expr) -> pl.Expr:
        return (
            pl.when(has_params.or_(has_varargs))
            .then(
                pl.concat_str(
                    pl.lit("\n\n    Args:\n"),
                    pl.col("param_doc_join"),
                    pl.when(has_varargs)
                    .then(
                        pl.concat_str(
                            pl.when(has_params)
                            .then(pl.lit("\n        "))
                            .otherwise(pl.lit("        ")),
                            pl.lit("*args ("),
                            pl.col("varargs_py_type"),
                            pl.lit("): `"),
                            varargs,
                            pl.lit("` expression"),
                        )
                    )
                    .otherwise(_EMPTY_STR),
                )
            )
            .otherwise(_EMPTY_STR)
        )

    def _body(has_params: pl.Expr) -> pl.Expr:
        return (
            pl.when(has_params.or_(has_varargs))
            .then(
                pl.concat_str(
                    pl.lit(", "),
                    pl.col("param_names_join"),
                    pl.when(has_varargs)
                    .then(
                        pl.when(has_params)
                        .then(pl.lit(", *args"))
                        .otherwise(pl.lit("*args"))
                    )
                    .otherwise(_EMPTY_STR),
                )
            )
            .otherwise(_EMPTY_STR)
        )

    return (
        expr.list.len()
        .cast(pl.Boolean)
        .pipe(
            lambda has_params: pl.concat_str(
                _signature(has_params),
                pl.lit("\n"),
                pl.lit('    """'),
                _description(),
                _args_section(has_params),
                pl.lit(f"\n\n    Returns:\n        {PyTypes.EXPR}: `"),
                pl.col("return_type").fill_null(DuckDbTypes.ANY.value.upper()),
                pl.lit('` expression.\n    """'),
                pl.lit("\n"),
                pl.lit('    return func("'),
                fn_name,
                pl.lit('"'),
                _body(has_params),
                pl.lit(")"),
            )
        )
    )
