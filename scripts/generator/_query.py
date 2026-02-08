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
    PyTypes,
)
from ._schemas import (
    DuckCols,
    FuncTypes,
    ParamLens,
    ParamLists,
    Params,
    PyCols,
)

_EMPTY_STR = pl.lit("")


def run_qry(lf: pl.LazyFrame) -> pl.LazyFrame:
    py = PyCols()
    params = Params()
    dk = DuckCols()
    return (
        lf.select(dk.to_dict().keys())
        .filter(
            dk.function_type.is_in(
                {FuncTypes.TABLE, FuncTypes.TABLE_MACRO, FuncTypes.PRAGMA}
            ).not_(),
            dk.parameters.list.len()
            .eq(0)
            .and_(dk.varargs.is_null())
            .not_(),  # literals
            dk.function_name.is_in(OPERATOR_MAP).not_(),
            *PREFIXES.iter().map(
                lambda prefix: dk.function_name.str.starts_with(prefix).not_()
            ),
            dk.function_name.ne("alias"),  # conflicts with duckdb alias method
            dk.alias_of.is_null().or_(dk.alias_of.is_in(OPERATOR_MAP)),
        )
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
        .with_row_index("sig_id")
        .explode("parameters", "parameter_types")
        .with_columns(
            pl.int_range(pl.len()).over("sig_id").alias("param_idx"),
            dk.parameters.pipe(_to_param_names, py.name),
        )
        .group_by(py.name, params.idx, maintain_order=True)
        .agg(
            pl.all().exclude("parameter_types").drop_nulls().first(),
            dk.parameter_types.pipe(_into_union),
            dk.parameter_types.pipe(_convert_duckdb_type_to_python)
            .pipe(_into_union)
            .pipe(_make_type_union)
            .alias("py_types"),
        )
        .group_by(py.name, maintain_order=True)
        .agg(
            pl.all().exclude("param_names").first(),
            *params.names.pipe(
                lambda expr: expr.filter(expr.is_not_null()).pipe(
                    _joined_parts,
                    params.idx.ge(params.lens.min_params_per_fn),
                    py.types,
                    dk.parameter_types,
                    py.namespace.pipe(_self_type),
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


def _into_union(expr: pl.Expr) -> pl.Expr:
    return expr.filter(expr.ne(_EMPTY_STR)).unique().sort().str.join(" | ")


def _joined_parts(
    expr: pl.Expr,
    cond: pl.Expr,
    py_union: pl.Expr,
    params_union: pl.Expr,
    self_type: pl.Expr,
) -> Iterable[pl.Expr]:
    py_type = _replace_self(py_union, self_type)

    def _param_sig_list() -> pl.Expr:
        return pl.concat_str(
            expr,
            pl.lit(": "),
            py_type,
            pl.when(cond).then(pl.lit(" | None = None")).otherwise(_EMPTY_STR),
        )

    def _param_doc_join() -> pl.Expr:
        return pl.concat_str(
            pl.lit("            "),
            expr,
            pl.lit(" ("),
            py_type,
            pl.when(cond).then(pl.lit(" | None")).otherwise(_EMPTY_STR),
            pl.lit("): `"),
            _replace_self(params_union, self_type),
            pl.lit("` expression"),
        ).str.join("\n")

    return (
        expr.alias("param_names_list"),
        expr.str.join(", ").alias("param_names_join"),
        _param_sig_list().alias("param_sig_list"),
        _param_doc_join().alias("param_doc_join"),
    )


def _make_type_union(py_type: pl.Expr) -> pl.Expr:
    self_value = pl.lit(PyTypes.SELF.value)
    return (
        pl.when(py_type.eq(_EMPTY_STR))
        .then(self_value)
        .otherwise(pl.concat_str(self_value, pl.lit(" | "), py_type))
    )


def _replace_self(expr: pl.Expr, self_type: pl.Expr) -> pl.Expr:
    return expr.str.replace_all("Self", self_type)


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


def _self_type(namespace: pl.Expr) -> pl.Expr:
    return pl.when(namespace.is_not_null()).then(pl.lit("T")).otherwise(pl.lit("Self"))


def _self_expr(namespace: pl.Expr) -> pl.Expr:
    return (
        pl.when(namespace.is_not_null())
        .then(pl.lit("self._parent.inner()"))
        .otherwise(pl.lit("self._expr"))
    )


def _convert_duckdb_type_to_python(param_type: pl.Expr) -> pl.Expr:
    return param_type.replace_strict(
        CONVERTER, default=PyTypes.SELF.value, return_dtype=pl.String
    )


def _to_param_names(params: pl.Expr, py_name: pl.Expr) -> pl.Expr:
    return (
        params.str.strip_chars_start("'\"[")
        .str.strip_chars_end("'\"[]")
        .str.replace(r"\(.*$", _EMPTY_STR)
        .str.replace_all(r"\.\.\.", _EMPTY_STR)
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
        dk.categories.cast(pl.List(pl.String()))
        .list.join("_")
        .pipe(
            lambda cat_str: (
                dk.function_name.pipe(
                    lambda expr: (
                        pl.when(expr.is_in(KWORDS))
                        .then(pl.concat_str(expr, pl.lit("_fn")))
                        .otherwise(expr)
                        .str.to_lowercase()
                    )
                )
                .pipe(
                    lambda base_name: (
                        pl.when(
                            cat_str.n_unique()
                            .over(dk.function_name)
                            .gt(1)
                            .and_(cat_str.ne(_EMPTY_STR))
                        )
                        .then(pl.concat_str(cat_str, pl.lit("_"), base_name))
                        .otherwise(base_name)
                    )
                )
                .pipe(
                    lambda base: (
                        pl.when(
                            dk.description.n_unique()
                            .over(dk.function_name, dk.categories)
                            .gt(1)
                            .and_(
                                p_lens.min_params_per_fn_cat_desc.gt(
                                    p_lens.min_params_per_fn
                                )
                            )
                        )
                        .then(
                            pl.concat_str(
                                base,
                                pl.lit("_"),
                                pl.when(
                                    p_lens.sig_param_count.eq(
                                        p_lens.min_params_per_fn_cat_desc
                                    )
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
                                .str.to_lowercase()
                                .max()
                                .over(dk.function_name, dk.categories, dk.description),
                            )
                        )
                        .otherwise(base)
                    )
                )
                .alias("py_name")
            )
        )
    )


def _to_func(
    has_params: pl.Expr, py: PyCols, p_lists: ParamLists, dk: DuckCols
) -> pl.Expr:
    self_type = py.namespace.pipe(_self_type)
    self_expr = py.namespace.pipe(_self_expr)
    varargs_type = (
        dk.varargs.pipe(_convert_duckdb_type_to_python)
        .pipe(_make_type_union)
        .pipe(_replace_self, self_type)
    )

    def _signature(has_params: pl.Expr) -> pl.Expr:
        return pl.concat_str(
            pl.lit("    def "),
            py.name,
            pl.lit("(self"),
            pl.when(has_params.gt(1))
            .then(
                pl.concat_str(
                    pl.lit(", "), p_lists.signatures.list.slice(1).list.join(", ")
                )
            )
            .otherwise(_EMPTY_STR),
            pl.when(dk.varargs.is_not_null())
            .then(pl.concat_str(pl.lit(", *args: "), varargs_type))
            .otherwise(_EMPTY_STR),
            pl.lit(") -> "),
            self_type,
            pl.lit(":"),
        )

    def _description() -> pl.Expr:
        return (
            pl.when(dk.description.is_not_null())
            .then(
                dk.description.str.strip_chars()
                .str.replace_all("\u2019", "'")
                .str.replace_all(r"\. ", ".\n\n        ")
                .str.strip_chars_end(".")
                .add(pl.lit("."))
            )
            .otherwise(
                pl.concat_str(pl.lit("SQL "), dk.function_name, pl.lit(" function."))
            )
        )

    def _args_section(has_params: pl.Expr) -> pl.Expr:
        return (
            pl.when(has_params.gt(1).or_(dk.varargs.is_not_null()))
            .then(
                pl.concat_str(
                    pl.lit("\n\n        Args:\n"),
                    p_lists.docs.str.split("\n").list.slice(1).list.join("\n"),
                    pl.when(dk.varargs.is_not_null())
                    .then(
                        pl.concat_str(
                            pl.lit("\n            *args ("),
                            varargs_type,
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

    def _return_cls() -> pl.Expr:
        return (
            pl.when(py.namespace.is_not_null())
            .then(pl.lit("self._parent.__class__"))
            .otherwise(pl.lit("self.__class__"))
        )

    def _body(has_params: pl.Expr) -> pl.Expr:
        slf_arg = pl.concat_str(pl.lit(", "), self_expr)
        sep = pl.lit(", ")
        args = p_lists.names.str.split(", ").list.slice(1).list.join(", ")

        def _reversed() -> pl.Expr:
            """Special case for log function: other args first, then self._expr."""
            return pl.when(dk.function_name.eq("log")).then(
                pl.concat_str(sep, args, slf_arg)
            )

        def _normal() -> pl.Expr:
            """Normal case: self._expr first, then other args."""
            return pl.concat_str(
                slf_arg,
                pl.when(has_params.gt(1))
                .then(pl.concat_str(sep, args))
                .otherwise(_EMPTY_STR),
            )

        return pl.concat_str(
            pl.when(dk.function_name.eq("log").and_(has_params.gt(1)))
            .then(_reversed())
            .otherwise(_normal()),
            pl.when(dk.varargs.is_not_null())
            .then(pl.lit(", *args"))
            .otherwise(_EMPTY_STR),
        )

    return pl.concat_str(
        _signature(has_params),
        pl.lit('\n        """'),
        _description(),
        _args_section(has_params),
        pl.lit("\n\n        Returns:\n            "),
        self_type,
        pl.lit('\n        """\n        return '),
        _return_cls(),
        pl.lit('(func("'),
        dk.function_name,
        pl.lit('"'),
        _body(has_params),
        pl.lit("))"),
    )
