from collections.abc import Callable, Iterable
from functools import partial
from typing import NamedTuple

import duckdb
import polars as pl

from ._models import (
    CATEGORY_RULES,
    CONVERSION_MAP,
    KWORDS,
    OPERATOR_MAP,
    SHADOWERS,
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


def get_df() -> pl.DataFrame:
    fn_name = pl.col("function_name")
    fn_type = pl.col("function_type")
    params = pl.col("parameters")
    py_name = pl.col("python_name")
    param_types = pl.col("parameter_types")
    param_names = pl.col("param_names")
    category = pl.col("category")
    description = pl.col("description")
    return_type = pl.col("return_type")
    varargs = pl.col("varargs")
    has_varargs = pl.col("has_varargs")
    has_params = pl.col("has_params")

    return (
        sql_query()
        .pl(lazy=True)
        .select(
            fn_name,
            fn_type.cast(FuncTypes),
            return_type,
            params,
            param_types.list.eval(pl.element().fill_null("ANY")),
            varargs,
            description,
            fn_name.pipe(_python_name),
            fn_name.pipe(_python_name).pipe(_category),
            params.list.len().alias("param_len"),
            params.list.len().min().over(fn_name).alias("min_param_len"),
        )
        .filter(
            fn_type.is_in(
                (FuncTypes.SCALAR, FuncTypes.AGGREGATE, FuncTypes.MACRO)
            ).and_(fn_name.str.starts_with("__").not_())
        )
        .sort(by=[fn_name, "param_len"], descending=[False, True])
        .unique(subset=py_name, keep="first")
        .with_columns(
            param_types.list.eval(
                pl.element().pipe(_convert_duckdb_type_to_python)
            ).alias("py_types"),
            varargs.pipe(_convert_duckdb_type_to_python).alias("varargs_py_type"),
        )
        .explode("parameters", "parameter_types", "py_types")
        .with_columns(
            params.pipe(_clean_param_name, py_name).alias("param_names"),
            params.cum_count().over(py_name).sub(1).alias("param_idx"),
            pl.col("py_types").str.contains(r"\bbool\b").alias("param_is_bool"),
        )
        .sort(by=[py_name, "param_idx"])
        .group_by(py_name, maintain_order=True)
        .agg(
            pl.all().exclude("param_names", "param_doc_join").first(),
            pl.col("param_is_bool")
            .filter(params.is_not_null())
            .alias("param_is_bool_list"),
            *param_names.filter(params.is_not_null()).pipe(
                _build_func_join,
                param_types,
                pl.col("py_types").pipe(_make_type_union),
            ),
        )
        .with_columns(
            varargs.is_not_null().alias("has_varargs"),
            pl.col("param_names_list").list.len().cast(pl.Boolean).alias("has_params"),
        )
        .with_columns(
            has_params.pipe(
                _build_varargs_parts,
                has_varargs,
                varargs,
                pl.col("varargs_py_type").pipe(_make_type_union),
            )
        )
        .select(
            category,
            py_name,
            SigParts(
                pl.col("param_sig_list"),
                pl.col("param_is_bool_list"),
                pl.col("sig_varargs_part"),
                has_varargs,
            ).into_expr(has_params.or_(has_varargs), py_name),
            pl.concat_str(
                pl.lit('    """'),
                description.pipe(_description_section, fn_name),
                has_params.or_(has_varargs).pipe(
                    _doc_args_section,
                    pl.col("param_doc_join"),
                    pl.col("doc_varargs_part"),
                ),
                return_type.pipe(_returns_section),
            ).alias("docstring"),
            has_params.or_(has_varargs).pipe(
                _get_body,
                fn_name,
                pl.col("param_names_join"),
                pl.col("body_varargs_part"),
            ),
        )
        .sort(category, py_name)
        .collect()
    )


def _make_type_union(py_type: pl.Expr) -> pl.Expr:
    return (
        pl.when(py_type.eq(PyTypes.EXPR.value))
        .then(py_type)
        .otherwise(pl.concat_str(pl.lit(PyTypes.EXPR.value), pl.lit(" | "), py_type))
    )


def _build_func_join(
    params_filter: pl.Expr, param_type: pl.Expr, py_type: pl.Expr
) -> Iterable[pl.Expr]:
    def _optional_suffix(suffix: str) -> pl.Expr:
        return (
            pl.when(pl.col("param_idx").ge(pl.col("min_param_len")))
            .then(pl.lit(suffix))
            .otherwise(_EMPTY_STR)
        )

    def _build_param_sig(expr: pl.Expr, py_type: pl.Expr) -> pl.Expr:
        return pl.concat_str(
            expr,
            pl.lit(": "),
            py_type,
            _optional_suffix(" | None"),
            _optional_suffix(" = None"),
        )

    def _build_param_doc(
        expr: pl.Expr, param_type: pl.Expr, py_type: pl.Expr
    ) -> pl.Expr:
        return pl.concat_str(
            pl.lit("        "),
            expr,
            pl.lit(" ("),
            py_type,
            _optional_suffix(" | None"),
            pl.lit("): `"),
            param_type,
            pl.lit("` expression"),
        )

    return (
        params_filter.alias("param_names_list"),
        params_filter.str.join(", ").alias("param_names_join"),
        params_filter.pipe(_build_param_sig, py_type).alias("param_sig_list"),
        params_filter.pipe(_build_param_doc, param_type, py_type)
        .str.join("\n")
        .alias("param_doc_join"),
    )


def _build_varargs_parts(
    has_params: pl.Expr,
    has_varargs: pl.Expr,
    varargs: pl.Expr,
    varargs_py_type: pl.Expr,
) -> Iterable[pl.Expr]:
    _if_params = pl.when(has_params).then

    def _sig() -> pl.Expr:
        return pl.concat_str(
            _if_params(pl.lit(", *args: ")).otherwise(pl.lit("*args: ")),
            varargs_py_type,
        )

    def _body() -> pl.Expr:
        return _if_params(pl.lit(", *args")).otherwise(pl.lit("*args"))

    def _doc() -> pl.Expr:
        return pl.concat_str(
            _if_params(pl.lit("\n        ")).otherwise(pl.lit("        ")),
            pl.lit("*args ("),
            varargs_py_type,
            pl.lit("): `"),
            varargs,
            pl.lit("` expression"),
        )

    def _build(expr_fn: Callable[[], pl.Expr]) -> pl.Expr:
        return (
            pl.when(has_varargs)
            .then(expr_fn())
            .otherwise(_EMPTY_STR)
            .alias(f"{expr_fn.__name__.removeprefix('_')}_varargs_part")
        )

    return (_build(_sig), _build(_body), _build(_doc))


def _convert_duckdb_type_to_python(param_type: pl.Expr) -> pl.Expr:
    return (
        param_type.str.extract(r"^([A-Z]+)", 1)
        .replace(
            CONVERSION_MAP.items()
            .iter()
            .map_star(lambda k, v: (k.value.upper(), v.value))
            .collect(dict),
            default=PyTypes.EXPR.value,
            return_dtype=pl.String,
        )
        .fill_null(PyTypes.EXPR.value)
    )


class SigParts(NamedTuple):
    param_sig_list: pl.Expr
    param_is_bool_list: pl.Expr
    sig_varargs: pl.Expr
    has_varargs: pl.Expr

    def _param_sig_with_kw(self) -> pl.Expr:
        has_bool = self.param_is_bool_list.list.any()
        first_bool_idx = (
            pl.when(has_bool)
            .then(self.param_is_bool_list.list.arg_max())
            .otherwise(pl.lit(0))
        )
        return (
            pl.when(has_bool.and_(self.has_varargs.not_()))
            .then(
                pl.concat_list(
                    self.param_sig_list.list.slice(0, first_bool_idx),
                    pl.lit(["*"]),
                    self.param_sig_list.list.slice(
                        first_bool_idx, self.param_sig_list.list.len()
                    ),
                )
            )
            .otherwise(self.param_sig_list)
            .list.join(", ")
        )

    def into_expr(self, cond: pl.Expr, py_name: pl.Expr) -> pl.Expr:
        _signature = partial(pl.concat_str, pl.lit("def "), py_name)

        return (
            pl.when(cond)
            .then(
                _signature(
                    pl.lit("("),
                    self._param_sig_with_kw(),
                    self.sig_varargs,
                    pl.lit(f") -> {PyTypes.EXPR}:"),
                )
            )
            .otherwise(_signature(pl.lit(f"() -> {PyTypes.EXPR}:")))
            .alias("signature")
        )


def _doc_args_section(
    cond: pl.Expr, param_doc: pl.Expr, doc_varargs: pl.Expr
) -> pl.Expr:
    return (
        pl.when(cond)
        .then(pl.concat_str(pl.lit("\n\n    Args:\n"), param_doc, doc_varargs))
        .otherwise(_EMPTY_STR)
        .alias("doc_args_section")
    )


def _description_section(description: pl.Expr, fn_name: pl.Expr) -> pl.Expr:
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


def _returns_section(return_type: pl.Expr) -> pl.Expr:
    return pl.concat_str(
        pl.lit(f"\n\n    Returns:\n        {PyTypes.EXPR}: `"),
        return_type.fill_null("ANY"),
        pl.lit('` expression.\n    """'),
    )


def _get_body(
    cond: pl.Expr, fn_name: pl.Expr, param_names: pl.Expr, body_varargs: pl.Expr
) -> pl.Expr:
    def _args_section() -> pl.Expr:
        return (
            pl.when(cond)
            .then(pl.concat_str(pl.lit(", "), param_names, body_varargs))
            .otherwise(_EMPTY_STR)
        )

    return pl.concat_str(
        pl.lit('    return func("'), fn_name, pl.lit('"'), _args_section(), pl.lit(")")
    ).alias("body")


def _clean_param_name(expr: pl.Expr, py_name: pl.Expr) -> pl.Expr:
    def _handle_python_keywords(expr: pl.Expr) -> pl.Expr:
        return (
            pl.when(expr.is_in(SHADOWERS))
            .then(pl.concat_str(expr, pl.lit("_arg")))
            .otherwise(expr)
        )

    def _keep_only_valid_identifiers(expr: pl.Expr) -> pl.Expr:
        return (
            pl.when(expr.str.contains(r"^[A-Za-z_][A-Za-z0-9_]*$").not_())
            .then(_EMPTY_STR)
            .otherwise(expr)
        )

    def _generate_fallback_names(expr: pl.Expr) -> pl.Expr:
        return (
            pl.when(expr.eq(_EMPTY_STR))
            .then(
                pl.concat_str(
                    pl.lit("arg"), expr.cum_count().over(py_name).cast(pl.String)
                )
            )
            .otherwise(expr)
        )

    def _deduplicate_within_function(expr: pl.Expr) -> pl.Expr:
        return (
            pl.when(expr.cum_count().over([py_name, expr]).gt(1))
            .then(
                pl.concat_str(
                    expr,
                    pl.lit("_"),
                    expr.cum_count().over(py_name, expr).cast(pl.String),
                )
            )
            .otherwise(expr)
        )

    return (
        expr.str.strip_chars_start("'\"[")
        .str.strip_chars_end("'\"[]")
        .str.replace(r"\(.*$", _EMPTY_STR)
        .str.replace_all(r"\.\.\.", _EMPTY_STR)
        .pipe(_handle_python_keywords)
        .pipe(_keep_only_valid_identifiers)
        .pipe(_generate_fallback_names)
        .pipe(_deduplicate_within_function)
    )


def _python_name(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr.is_in(OPERATOR_MAP.keys()))
        .then(expr.replace(OPERATOR_MAP, default=expr, return_dtype=pl.String))
        .when(expr.is_in(KWORDS))
        .then(pl.concat_str(expr, pl.lit("_func")))
        .otherwise(expr)
        .str.replace_all(r"([a-z0-9])([A-Z])", r"$1_$2")
        .str.to_lowercase()
        .alias("python_name")
    )


def _category(expr: pl.Expr) -> pl.Expr:
    return pl.coalesce(
        CATEGORY_RULES.iter().map(lambda cat: cat.into_expr(expr))
    ).alias("category")
