from collections.abc import Iterable

import polars as pl

from ._rules import (
    CONVERTER,
    KWORDS,
    NAMESPACE_SPECS,
    OPERATOR_MAP,
    PREFIXES,
    SHADOWERS,
    DuckDbTypes,
)
from ._schemas import (
    DuckCols,
    FuncTypes,
    ParamLens,
    ParamLists,
    Params,
    PyCols,
)
from ._str_builder import EMPTY_STR, format_kwords


def run_qry(lf: pl.LazyFrame) -> pl.LazyFrame:
    py = PyCols()
    params = Params()
    dk = DuckCols()
    return (
        lf.select(dk.to_dict().keys())
        .filter(_filters(dk))
        .with_columns(
            dk.parameter_types.list.eval(pl.element().fill_null(DuckDbTypes.ANY)),
            *dk.parameters.list.len().pipe(
                lambda expr_len: (
                    expr_len.alias("sig_param_count"),
                    expr_len.min().over(dk.function_name).alias("min_params_per_fn"),
                    expr_len.min()
                    .over(dk.function_name, dk.categories, dk.description)
                    .alias("min_params_per_fn_cat_desc"),
                )
            ),
        )
        .with_columns(_to_py_name(dk, params.lens))
        .with_columns(dk.categories.pipe(_namespace_specs, py.name))
        .with_columns(
            pl.when(py.namespace.is_not_null())
            .then(pl.lit("T"))
            .otherwise(pl.lit("Self"))
            .alias("self_type")
        )
        .with_row_index("sig_id")
        .explode("parameters", "parameter_types")
        .with_columns(
            pl.int_range(pl.len()).over("sig_id").alias("param_idx"),
            dk.parameters.pipe(_to_param_names, py.name),
        )
        .group_by(py.namespace, py.name, params.idx, maintain_order=True)
        .agg(
            pl.all().exclude("parameter_types").drop_nulls().first(),
            dk.parameter_types.pipe(_into_union),
            dk.parameter_types.pipe(_convert_duckdb_type_to_python)
            .pipe(_into_union)
            .pipe(_make_type_union, py.self_type.first())
            .alias("py_types"),
        )
        .group_by(py.namespace, py.name, maintain_order=True)
        .agg(
            pl.all().exclude("param_names").first(),
            *params.names.pipe(
                lambda expr: expr.pipe(
                    _joined_parts,
                    params.idx.ge(params.lens.min_params_per_fn),
                    py.types,
                    dk.parameter_types,
                )
            ),
        )
        .select(
            py.namespace,
            py.name,
            pl.col("param_names_list").list.len().pipe(_to_func, py, ParamLists(), dk),
        )
        .sort(py.namespace, py.name)
    )


def _filters(dk: DuckCols) -> Iterable[pl.Expr]:
    return (
        dk.function_type.is_in(
            {FuncTypes.TABLE, FuncTypes.TABLE_MACRO, FuncTypes.PRAGMA}
        ).not_(),
        dk.parameters.list.len().eq(0).and_(dk.varargs.is_null()).not_(),  # literals
        dk.function_name.is_in(OPERATOR_MAP).not_(),
        *PREFIXES.iter().map(
            lambda prefix: dk.function_name.str.starts_with(prefix).not_()
        ),
        dk.function_name.ne("alias"),  # conflicts with duckdb alias method
        dk.alias_of.is_null().or_(dk.alias_of.is_in(OPERATOR_MAP)),
    )


def _into_union(expr: pl.Expr) -> pl.Expr:
    return expr.filter(expr.ne(EMPTY_STR)).unique().sort().str.join(" | ")


def _joined_parts(
    expr: pl.Expr, cond: pl.Expr, py_union: pl.Expr, params_union: pl.Expr
) -> Iterable[pl.Expr]:
    def _param_sig_list() -> pl.Expr:

        return format_kwords(
            "{param_name}: {py_type}{union}",
            param_name=expr,
            py_type=py_union,
            union=pl.when(cond).then(pl.lit(" | None = None")),
            ignore_nulls=True,
        )

    def _param_doc_list() -> pl.Expr:
        return format_kwords(
            "            {param_name} ({py_type}{union}): `{dk_type}` expression",
            param_name=expr,
            py_type=py_union,
            union=pl.when(cond).then(pl.lit(" | None")),
            dk_type=params_union,
            ignore_nulls=True,
        )

    return (
        expr.alias("param_names_list"),
        _param_sig_list().alias("param_sig_list"),
        _param_doc_list().alias("param_doc_list"),
    )


def _make_type_union(py_type: pl.Expr, self_type: pl.Expr) -> pl.Expr:
    txt = "{self_type} | {py_type}"

    return (
        pl.when(py_type.eq(EMPTY_STR))
        .then(self_type)
        .otherwise(format_kwords(txt, self_type=self_type, py_type=py_type))
    )


def _namespace_specs(cats: pl.Expr, py_name: pl.Expr) -> pl.Expr:

    return pl.coalesce(
        *NAMESPACE_SPECS.iter().map(
            lambda spec: pl.when(
                spec.categories.iter()
                .map(lambda c: c.value)
                .into(lambda x: cats.list.set_intersection(x.collect(tuple)))
                .list.len()
                .gt(0)
            ).then(pl.lit(spec.name))
        ),
        *NAMESPACE_SPECS.iter().map(
            lambda spec: pl.when(
                spec.prefixes.iter()
                .map(py_name.str.starts_with)
                .fold(pl.lit(value=False), lambda a, b: a.or_(b))
            ).then(pl.lit(spec.name))
        ),
    ).alias("namespace")


def _convert_duckdb_type_to_python(param_type: pl.Expr) -> pl.Expr:
    return param_type.replace_strict(CONVERTER, return_dtype=pl.String)


def _to_param_names(params: pl.Expr, py_name: pl.Expr) -> pl.Expr:
    return (
        params.str.strip_chars_start("'\"[")
        .str.strip_chars_end("'\"[]")
        .str.replace(r"\(.*$", EMPTY_STR)
        .str.replace_all(r"\.\.\.", EMPTY_STR)
        .pipe(
            lambda expr: (
                pl.when(expr.is_in(SHADOWERS))
                .then(pl.concat_str(expr, pl.lit("_arg")))
                .otherwise(expr)
            )
        )
        .pipe(
            lambda expr: (
                pl.when(expr.cum_count().over(py_name, expr).gt(1))
                .then(
                    pl.concat_str(
                        expr,
                        pl.lit("_"),
                        expr.cum_count().over(py_name, expr).cast(pl.String),
                    )
                )
                .otherwise(expr)
            )
        )
        .alias("param_names")
    )


def _to_py_name(dk: DuckCols, p_lens: ParamLens) -> pl.Expr:
    return (
        dk.function_name.pipe(
            lambda expr: (
                pl.when(expr.is_in(KWORDS))
                .then(pl.concat_str(expr, pl.lit("_fn")))
                .otherwise(expr)
            )
        )
        .pipe(
            lambda base: (
                pl.when(
                    dk.description.n_unique()
                    .over(dk.function_name, dk.categories)
                    .gt(1)
                    .and_(
                        p_lens.min_params_per_fn_cat_desc.gt(p_lens.min_params_per_fn)
                    )
                )
                .then(
                    pl.concat_str(
                        base,
                        pl.lit("_"),
                        pl.when(
                            p_lens.sig_param_count.eq(p_lens.min_params_per_fn_cat_desc)
                        )
                        .then(
                            pl.when(
                                p_lens.min_params_per_fn_cat_desc.eq(
                                    p_lens.min_params_per_fn
                                )
                            )
                            .then(dk.parameters)
                            .otherwise(
                                dk.parameters.list.slice(
                                    p_lens.min_params_per_fn,
                                    p_lens.sig_param_count.sub(
                                        p_lens.min_params_per_fn
                                    ),
                                )
                            )
                        )
                        .otherwise(pl.lit([], dtype=pl.List(pl.String)))
                        .list.join("_")
                        .max()
                        .over(dk.function_name, dk.categories, dk.description),
                    )
                )
                .otherwise(base)
            )
        )
        .str.to_lowercase()
        .alias("py_name")
    )


def _to_func(
    has_params: pl.Expr, py: PyCols, p_lists: ParamLists, dk: DuckCols
) -> pl.Expr:

    varargs_type = dk.varargs.pipe(_convert_duckdb_type_to_python).pipe(
        _make_type_union, py.self_type
    )

    def _duckdb_args(has_params: pl.Expr) -> pl.Expr:
        def _self_expr() -> pl.Expr:
            return format_kwords(
                ", self.{expr}",
                expr=pl.when(py.namespace.is_not_null())
                .then(pl.lit("_parent.inner()"))
                .otherwise(pl.lit("_expr")),
            )

        slf_arg = _self_expr()
        sep = pl.lit(", ")
        args = p_lists.names.list.slice(1).list.join(", ")

        def _reversed() -> pl.Expr:
            """Special case for log function: other args first, then self._expr."""
            return pl.when(dk.function_name.eq("log")).then(
                pl.concat_str(sep, args, slf_arg)
            )

        return (
            pl.when(dk.function_name.eq("log").and_(has_params.gt(1)))
            .then(_reversed())
            .otherwise(
                pl.concat_str(
                    slf_arg,
                    pl.when(has_params.gt(1)).then(pl.concat_str(sep, args)),
                    ignore_nulls=True,
                )
            )
        )

    return format_kwords(
        _txt(),
        func_name=py.name.alias("func_name"),
        args=pl.when(has_params.gt(1)).then(
            pl.format(", {}", p_lists.signatures.list.slice(1).list.join(", "))
        ),
        varargs=pl.when(dk.varargs.is_not_null()).then(
            pl.format(", *args: {}", varargs_type)
        ),
        self_type=py.self_type,
        description=pl.when(dk.description.is_not_null())
        .then(
            dk.description.str.strip_chars()
            .str.replace_all("\u2019", "'")
            .str.replace_all(r"\. ", ".\n\n        ")
            .str.strip_chars_end(".")
        )
        .otherwise(pl.format("SQL {} function", dk.function_name)),
        args_section=pl.when(has_params.gt(1).or_(dk.varargs.is_not_null())).then(
            format_kwords(
                "\n\n        Args:\n{posargs}{varargs}",
                posargs=p_lists.docs.list.slice(1).list.join("\n"),
                varargs=pl.when(dk.varargs.is_not_null()).then(
                    format_kwords(
                        "\n            *args ({pytypes}): `{ducktypes}` expression",
                        pytypes=varargs_type,
                        ducktypes=dk.varargs,
                    )
                ),
                ignore_nulls=True,
            )
        ),
        parent=pl.when(py.namespace.is_not_null()).then(pl.lit("_parent.")),
        duckdb_name=dk.function_name,
        duckdb_args=has_params.pipe(_duckdb_args),
        duckdb_varargs=pl.when(dk.varargs.is_not_null()).then(pl.lit(", *args")),
        ignore_nulls=True,
    )


def _txt() -> str:
    return '''
    def {func_name}(self{args}{varargs}) -> {self_type}:
        """{description}.{args_section}

        Returns:
            {self_type}
        """
        return self.{parent}__class__(func("{duckdb_name}"{duckdb_args}{duckdb_varargs}))
        '''
