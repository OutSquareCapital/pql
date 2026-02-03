from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial

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

# TODO: ensure that all overloads (params) are taken into account. E.g regexp_extract -> same category, different params, different description
_EMPTY_STR = pl.lit("")


def sql_query() -> duckdb.DuckDBPyRelation:
    qry = """--sql
        SELECT *
        FROM duckdb_functions()
    """
    return duckdb.sql(qry)


def get_df() -> pl.LazyFrame:
    fn_name = pl.col("function_name")
    params = pl.col("parameters")
    py_name = pl.col("python_name")
    py_types = pl.col("py_types")
    param_types = pl.col("parameter_types")
    param_names = pl.col("param_names")
    category = pl.col("category")
    description = pl.col("description")
    return_type = pl.col("return_type")
    varargs = pl.col("varargs")
    has_varargs = pl.col("has_varargs")
    has_params = pl.col("has_params")

    duckdb_cats = pl.col("categories")
    fn_name = pl.col("function_name")
    alias_of = pl.col("alias_of")

    return (
        sql_query()
        .pl(lazy=True)
        .filter(
            pl.col("function_type")
            .cast(FuncTypes)
            .is_in((FuncTypes.SCALAR, FuncTypes.AGGREGATE, FuncTypes.MACRO))
            .and_(fn_name.str.starts_with("__").not_())
            .and_(fn_name.is_in(OPERATOR_MAP).not_())
            .and_(alias_of.is_null().or_(alias_of.is_in(OPERATOR_MAP)))
        )
        .with_columns(
            params.list.len().alias("param_len"),
            params.list.len().min().over(fn_name).alias("min_param_len"),
            params.list.len()
            .min()
            .over(fn_name, duckdb_cats, description)
            .alias("min_param_len_desc"),
        )
        .with_columns(
            fn_name.pipe(
                _py_name,
                duckdb_cats,
                params.pipe(
                    _param_suffix,
                    pl.col("param_len"),
                    pl.col("min_param_len"),
                    pl.col("min_param_len_desc"),
                ).alias("param_suffix"),
                description.n_unique()
                .over(fn_name, duckdb_cats)
                .gt(1)
                .and_(pl.col("min_param_len_desc").gt(pl.col("min_param_len")))
                .alias("should_suffix"),
            )
        )
        .select(
            fn_name,
            return_type,
            params,
            param_types.list.eval(pl.element().fill_null("ANY")),
            varargs,
            description,
            py_name,
            py_name.pipe(_category),
            pl.col("param_len"),
            pl.col("min_param_len"),
        )
        .with_row_index("sig_id")
        .explode("parameters", "parameter_types")
        .with_columns(
            pl.int_range(pl.len()).over("sig_id").alias("param_idx"),
            param_types.pipe(_convert_duckdb_type_to_python).alias("py_types"),
            varargs.pipe(_convert_duckdb_type_to_python).alias("varargs_py_type"),
        )
        .with_columns(
            params.pipe(_clean_param_name, py_name).alias("param_names"),
            py_types.str.contains(r"\bbool\b").alias("param_is_bool"),
        )
        .group_by(py_name, "param_idx", maintain_order=True)
        .agg(
            pl.all()
            .exclude(
                "param_names",
                "param_doc_join",
                "py_types",
                "parameter_types",
                "param_is_bool",
            )
            .first(),
            param_names.drop_nulls().first(),
            param_types.drop_nulls()
            .unique()
            .str.join(" | ")
            .alias("param_types_union"),
            py_types.drop_nulls().unique().str.join(" | ").alias("py_types_union"),
            pl.col("param_is_bool").drop_nulls().first(),
        )
        .sort(by=[py_name, "param_idx"])
        .group_by(py_name, maintain_order=True)
        .agg(
            pl.all().exclude("param_names", "param_doc_join").first(),
            pl.col("param_is_bool")
            .filter(param_names.is_not_null())
            .alias("param_is_bool_list"),
            *FuncJoinParts(
                param_names.filter(param_names.is_not_null()),
                pl.col("param_types_union"),
                pl.col("py_types_union").pipe(_make_type_union),
            ).build(),
        )
        .with_columns(
            varargs.is_not_null().alias("has_varargs"),
            pl.col("param_names_list").list.len().cast(pl.Boolean).alias("has_params"),
        )
        .with_columns(
            *VarArgsParts(
                has_varargs,
                has_params,
                varargs,
                pl.col("varargs_py_type").pipe(_make_type_union),
            ).build()
        )
        .select(
            category,
            py_name,
            SigParts(has_varargs).build(has_params.or_(has_varargs), py_name),
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
    )


def _make_type_union(py_type: pl.Expr) -> pl.Expr:
    expr_val = pl.lit(PyTypes.EXPR.value)
    return (
        pl.when(py_type.eq(expr_val))
        .then(py_type)
        .otherwise(pl.concat_str(expr_val, pl.lit(" | "), py_type))
    )


@dataclass(slots=True)
class FuncJoinParts:
    params_filter: pl.Expr
    param_type: pl.Expr
    py_type: pl.Expr

    def _optional_suffix(self, suffix: str) -> pl.Expr:
        return (
            pl.when(pl.col("param_idx").ge(pl.col("min_param_len")))
            .then(pl.lit(suffix))
            .otherwise(_EMPTY_STR)
        )

    def _build_param_sig(self, expr: pl.Expr, py_type: pl.Expr) -> pl.Expr:
        return pl.concat_str(
            expr,
            pl.lit(": "),
            py_type,
            self._optional_suffix(" | None"),
            self._optional_suffix(" = None"),
        )

    def _build_param_doc(
        self, expr: pl.Expr, param_type: pl.Expr, py_type: pl.Expr
    ) -> pl.Expr:
        return pl.concat_str(
            pl.lit("        "),
            expr,
            pl.lit(" ("),
            py_type,
            self._optional_suffix(" | None"),
            pl.lit("): `"),
            param_type,
            pl.lit("` expression"),
        )

    def build(self) -> Iterable[pl.Expr]:
        return (
            self.params_filter.alias("param_names_list"),
            self.params_filter.str.join(", ").alias("param_names_join"),
            self.params_filter.pipe(self._build_param_sig, self.py_type).alias(
                "param_sig_list"
            ),
            self.params_filter.pipe(
                self._build_param_doc, self.param_type, self.py_type
            )
            .str.join("\n")
            .alias("param_doc_join"),
        )


@dataclass(slots=True)
class VarArgsParts:
    has_varargs: pl.Expr
    has_params: pl.Expr
    values: pl.Expr
    py_type: pl.Expr

    @property
    def _if_params(self):  # noqa: ANN202
        return pl.when(self.has_params).then

    def _sig(self) -> pl.Expr:
        return pl.concat_str(
            self._if_params(pl.lit(", *args: ")).otherwise(pl.lit("*args: ")),
            self.py_type,
        )

    def _body(self) -> pl.Expr:
        return self._if_params(pl.lit(", *args")).otherwise(pl.lit("*args"))

    def _doc(self) -> pl.Expr:
        return pl.concat_str(
            self._if_params(pl.lit("\n        ")).otherwise(pl.lit("        ")),
            pl.lit("*args ("),
            self.py_type,
            pl.lit("): `"),
            self.values,
            pl.lit("` expression"),
        )

    def _get_expr(self, expr_fn: Callable[[], pl.Expr]) -> pl.Expr:
        return (
            pl.when(self.has_varargs)
            .then(expr_fn())
            .otherwise(_EMPTY_STR)
            .alias(f"{expr_fn.__name__.removeprefix('_')}_varargs_part")
        )

    def build(self) -> Iterable[pl.Expr]:
        return (
            self._get_expr(self._sig),
            self._get_expr(self._body),
            self._get_expr(self._doc),
        )


def _convert_duckdb_type_to_python(param_type: pl.Expr) -> pl.Expr:
    converter = (
        CONVERSION_MAP.items()
        .iter()
        .map_star(lambda k, v: (k.value.upper(), v.value))
        .collect(dict)
    )
    val = PyTypes.EXPR.value
    return (
        param_type.str.extract(r"^([A-Z]+)", 1)
        .replace_strict(converter, default=val, return_dtype=pl.String)
        .fill_null(val)
    )


@dataclass(slots=True)
class SigParts:
    has_varargs: pl.Expr
    param_sig_list: pl.Expr = field(default=pl.col("param_sig_list"))
    param_is_bool_list: pl.Expr = field(default=pl.col("param_is_bool_list"))
    sig_varargs: pl.Expr = field(default=pl.col("sig_varargs_part"))

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

    def build(self, cond: pl.Expr, py_name: pl.Expr) -> pl.Expr:
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
    arg = pl.lit("_arg")

    def _handle_python_keywords(expr: pl.Expr) -> pl.Expr:
        return (
            pl.when(expr.is_in(SHADOWERS))
            .then(pl.concat_str(expr, arg))
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
            .then(pl.concat_str(arg, expr.cum_count().over(py_name).cast(pl.String)))
            .otherwise(expr)
        )

    def _deduplicate_within_function(expr: pl.Expr) -> pl.Expr:
        enumerated = pl.concat_str(
            expr, pl.lit("_"), expr.cum_count().over(py_name, expr).cast(pl.String)
        )
        return (
            pl.when(expr.cum_count().over([py_name, expr]).gt(1))
            .then(enumerated)
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


def _param_suffix(
    params: pl.Expr,
    param_len: pl.Expr,
    min_param_len: pl.Expr,
    min_param_len_desc: pl.Expr,
) -> pl.Expr:
    return (
        pl.when(param_len.eq(min_param_len_desc))
        .then(
            pl.when(min_param_len_desc.eq(min_param_len))
            .then(params)
            .otherwise(params.list.slice(min_param_len, param_len.sub(min_param_len)))
        )
        .otherwise(pl.lit([]))
        .list.eval(pl.element().cast(pl.String))
        .list.join("_")
        .str.to_lowercase()
        .str.replace_all(r"[^A-Za-z0-9_]+", "_")
        .str.replace_all(r"_+", "_")
        .str.strip_chars("_")
        .max()
        .over(pl.col("function_name"), pl.col("categories"), pl.col("description"))
    )


def _py_name(
    fn_name: pl.Expr,
    duckdb_cats: pl.Expr,
    param_suffix: pl.Expr,
    should_suffix: pl.Expr,
) -> pl.Expr:
    def _python_name(expr: pl.Expr) -> pl.Expr:
        return (
            pl.when(expr.is_in(KWORDS))
            .then(pl.concat_str(expr, pl.lit("_func")))
            .otherwise(expr)
            .str.replace_all(r"([a-z0-9])([A-Z])", r"$1_$2")
            .str.to_lowercase()
        )

    cat_str = duckdb_cats.list.join("_").fill_null(_EMPTY_STR).alias("cats_str")

    return (
        fn_name.pipe(_python_name)
        .pipe(
            lambda base_name: pl.when(
                cat_str.n_unique().over(fn_name).gt(1).and_(cat_str.ne(_EMPTY_STR))
            )
            .then(pl.concat_str(base_name, pl.lit("_"), cat_str))
            .otherwise(base_name)
        )
        .pipe(
            lambda base: pl.when(should_suffix)
            .then(pl.concat_str(base, pl.lit("_"), param_suffix))
            .otherwise(base)
        )
        .alias("python_name")
    )


def _category(expr: pl.Expr) -> pl.Expr:
    return pl.coalesce(
        CATEGORY_RULES.iter().map(lambda cat: cat.into_expr(expr))
    ).alias("category")
