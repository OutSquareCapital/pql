from collections.abc import Iterable
from dataclasses import dataclass, field

import duckdb
import polars as pl

from ._models import (
    CATEGORY_RULES,
    CONVERSION_MAP,
    KWORDS,
    OPERATOR_MAP,
    SHADOWERS,
    DuckDbTypes,
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


@dataclass(slots=True)
class PyCols:
    name: pl.Expr = field(default=pl.col("py_name"))
    types: pl.Expr = field(default=pl.col("py_types"))
    types_union: pl.Expr = field(default=pl.col("py_types_union"))


@dataclass(slots=True)
class Params:
    names: pl.Expr = field(default=pl.col("param_names"))
    idx: pl.Expr = field(default=pl.col("param_idx"))
    lens: ParamLens = field(default_factory=ParamLens)
    types_union: pl.Expr = field(default=pl.col("param_types_union"))


@dataclass(slots=True)
class DuckCols:
    name: pl.Expr = field(default=pl.col("function_name"))
    description: pl.Expr = field(default=pl.col("description"))
    category: pl.Expr = field(default=pl.col("category"))
    cats: pl.Expr = field(default=pl.col("categories"))
    varargs: pl.Expr = field(default=pl.col("varargs"))
    alias_of: pl.Expr = field(default=pl.col("alias_of"))
    params: pl.Expr = field(default=pl.col("parameters"))
    params_types: pl.Expr = field(default=pl.col("parameter_types"))


def get_df() -> pl.LazyFrame:
    py = PyCols()
    params = Params()
    dk = DuckCols()

    return (
        sql_query()
        .pl()
        .filter(
            dk.name.str.starts_with("__")
            .not_()
            .and_(dk.name.is_in(OPERATOR_MAP).not_())
            .and_(dk.alias_of.is_null().or_(dk.alias_of.is_in(OPERATOR_MAP)))
        )
        .with_columns(
            dk.params_types.list.eval(
                pl.element().fill_null(DuckDbTypes.ANY.value.upper())
            ),
            *dk.params.list.len().pipe(
                lambda expr_len: (
                    expr_len.alias("p_len_by_func"),
                    expr_len.min().over(dk.name).alias("p_len_by_func_and_cat"),
                    expr_len.min()
                    .over(dk.name, dk.cats, dk.description)
                    .alias("p_len_by_func_cat_and_desc"),
                )
            ),
            dk.varargs.pipe(_convert_duckdb_type_to_python)
            .pipe(_make_type_union)
            .alias("varargs_py_type"),
        )
        .with_columns(_to_py_name(dk, params.lens))
        .with_row_index("sig_id")
        .explode("parameters", "parameter_types")
        .with_columns(
            pl.int_range(pl.len()).over("sig_id").alias("param_idx"),
            dk.params_types.pipe(_convert_duckdb_type_to_python).alias("py_types"),
            dk.params.pipe(_to_param_names, py.name),
        )
        .group_by(py.name, params.idx, maintain_order=True)
        .agg(
            pl.all()
            .exclude("param_doc_join", "py_types", "parameter_types")
            .drop_nulls()
            .first(),
            dk.params_types.unique().str.join(" | ").alias("param_types_union"),
            py.types.unique().str.join(" | ").alias("py_types_union"),
        )
        .sort(py.name, params.idx)
        .group_by(py.name, maintain_order=True)
        .agg(
            pl.all().exclude("param_names", "param_doc_join").first(),
            *params.names.filter(params.names.is_not_null()).pipe(
                _joined_parts,
                params.idx.ge(params.lens.by_func_and_cat),
                py.types_union,
                params.types_union,
            ),
        )
        .select(
            pl.coalesce(
                CATEGORY_RULES.iter().map(lambda cat: cat.into_expr(py.name))
            ).alias("category"),
            py.name,
            pl.col("param_names_list")
            .list.len()
            .cast(pl.Boolean)
            .pipe(
                _to_func,
                py.name,
                pl.col("param_sig_list"),
                pl.col("param_doc_join"),
                pl.col("param_names_join"),
                pl.col("varargs_py_type"),
                pl.col("return_type"),
                dk,
            ),
        )
        .sort(dk.category, py.name)
        .lazy()
    )


def _joined_parts(
    expr: pl.Expr, cond: pl.Expr, _py_type_union: pl.Expr, param_types_union: pl.Expr
) -> Iterable[pl.Expr]:
    def _param_sig_list() -> pl.Expr:
        return pl.concat_str(
            expr,
            pl.lit(": "),
            _py_type_union.pipe(_make_type_union),
            pl.when(cond).then(pl.lit(" | None = None")).otherwise(_EMPTY_STR),
        ).alias("param_sig_list")

    def _param_doc_join() -> pl.Expr:
        return (
            pl.concat_str(
                pl.lit("        "),
                expr,
                pl.lit(" ("),
                _py_type_union.pipe(_make_type_union),
                pl.when(cond).then(pl.lit(" | None")).otherwise(_EMPTY_STR),
                pl.lit("): `"),
                param_types_union,
                pl.lit("` expression"),
            )
            .str.join("\n")
            .alias("param_doc_join")
        )

    return (
        expr.alias("param_names_list"),
        expr.str.join(", ").alias("param_names_join"),
        _param_sig_list(),
        _param_doc_join(),
    )


def _make_type_union(py_type: pl.Expr) -> pl.Expr:
    return (
        pl.when(py_type.eq(pl.lit(PyTypes.EXPR.value)))
        .then(py_type)
        .otherwise(pl.concat_str(pl.lit(f"{PyTypes.EXPR.value} | "), py_type))
    )


def _convert_duckdb_type_to_python(param_type: pl.Expr) -> pl.Expr:
    def _replace(pattern: str) -> pl.Expr:
        return param_type.str.extract(pattern, 1).replace_strict(
            CONVERSION_MAP.items()
            .iter()
            .map_star(lambda k, v: (k.value.upper(), v.value))
            .collect(dict),
            default=PyTypes.EXPR.value,
            return_dtype=pl.String,
        )

    return (
        pl.when(param_type.str.contains(r"\[\]$"))
        .then(pl.concat_str(pl.lit("list["), _replace(r"^([A-Z]+)\[\]$"), pl.lit("]")))
        .otherwise(_replace(r"^([A-Z]+)"))
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


def _to_py_name(dk: DuckCols, p_lens: ParamLens) -> pl.Expr:
    return dk.cats.list.join("_").pipe(
        lambda cat_str: dk.name.pipe(
            lambda expr: (
                pl.when(expr.is_in(KWORDS))
                .then(pl.concat_str(expr, pl.lit("_fn")))
                .otherwise(expr)
                .str.to_lowercase()
            )
        )
        .pipe(
            lambda base_name: pl.when(
                cat_str.n_unique().over(dk.name).gt(1).and_(cat_str.ne(_EMPTY_STR))
            )
            .then(pl.concat_str(base_name, pl.lit("_"), cat_str))
            .otherwise(base_name)
        )
        .pipe(
            lambda base: pl.when(
                dk.description.n_unique()
                .over(dk.name, dk.cats)
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
                        .then(dk.params)
                        .otherwise(
                            dk.params.list.slice(
                                p_lens.by_func_and_cat,
                                p_lens.by_func.sub(p_lens.by_func_and_cat),
                            )
                        )
                    )
                    .otherwise(pl.lit([], dtype=pl.List(pl.String)))
                    .list.join("_")
                    .str.to_lowercase()
                    .max()
                    .over(dk.name, dk.cats, dk.description),
                )
            )
            .otherwise(base)
        )
        .alias("py_name")
    )


def _to_func(
    has_params: pl.Expr,
    py_name: pl.Expr,
    param_sig_list: pl.Expr,
    param_doc_join: pl.Expr,
    param_names_join: pl.Expr,
    varargs_py_type: pl.Expr,
    return_type: pl.Expr,
    dk: DuckCols,
) -> pl.Expr:
    def _signature(has_params: pl.Expr) -> pl.Expr:
        return pl.concat_str(
            pl.when(has_params.or_(dk.varargs.is_not_null()))
            .then(
                pl.concat_str(
                    pl.lit("def "),
                    py_name,
                    pl.lit("("),
                    param_sig_list.list.join(", "),
                    pl.when(dk.varargs.is_not_null())
                    .then(
                        pl.concat_str(
                            pl.when(has_params)
                            .then(pl.lit(", *args: "))
                            .otherwise(pl.lit("*args: ")),
                            varargs_py_type,
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
            pl.when(dk.description.is_not_null())
            .then(
                dk.description.str.strip_chars()
                .str.replace_all("\u2019", "'")
                .str.replace_all(r"\. ", ".\n\n    ")
                .str.strip_chars_end(".")
                .add(pl.lit("."))
            )
            .otherwise(pl.concat_str(pl.lit("SQL "), dk.name, pl.lit(" function.")))
        )

    def _args_section(has_params: pl.Expr) -> pl.Expr:
        return (
            pl.when(has_params.or_(dk.varargs.is_not_null()))
            .then(
                pl.concat_str(
                    pl.lit("\n\n    Args:\n"),
                    param_doc_join,
                    pl.when(dk.varargs.is_not_null())
                    .then(
                        pl.concat_str(
                            pl.when(has_params)
                            .then(pl.lit("\n        "))
                            .otherwise(pl.lit("        ")),
                            pl.lit("*args ("),
                            varargs_py_type,
                            pl.lit("): `"),
                            dk.varargs,
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
            pl.when(has_params.or_(dk.varargs.is_not_null()))
            .then(
                pl.concat_str(
                    pl.lit(", "),
                    param_names_join,
                    pl.when(dk.varargs.is_not_null())
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

    return pl.concat_str(
        _signature(has_params),
        pl.lit('\n    """'),
        _description(),
        _args_section(has_params),
        pl.lit(f"\n\n    Returns:\n        {PyTypes.EXPR}: `"),
        return_type.fill_null(DuckDbTypes.ANY.value.upper()),
        pl.lit('` expression.\n    """\n    return func("'),
        dk.name,
        pl.lit('"'),
        _body(has_params),
        pl.lit(")"),
    )
