from collections.abc import Iterable

import polars as pl
import pyochain as pc

from ._rules import (
    CONVERTER,
    GENERIC_FUNCTIONS,
    NAMESPACE_SPECS,
    PREFIXES,
    SHADOWERS,
    SPECIAL_CASES,
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
        lf.pipe(_filters, dk)
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
        .pipe(lambda lf: lf.join(_alias_map(lf, dk), on=dk.function_name, how="left"))
        .pipe(
            lambda lf: lf.join(
                _py_name_map(lf, dk, params.lens),
                on=[dk.function_name, dk.categories, dk.description],
                how="left",
            )
        )
        .with_columns(dk.categories.pipe(_namespace_specs, dk.function_name))
        .explode("namespace")
        .with_columns(
            _py_name(dk, py),
            pl.when(py.namespace.is_not_null())
            .then(pl.lit("T"))
            .otherwise(pl.lit("Self"))
            .alias("self_type"),
        )
        .with_row_index("sig_id")
        .explode("parameters", "parameter_types")
        .with_columns(
            pl.int_range(pl.len()).over("sig_id").alias("param_idx"),
            dk.parameters.pipe(_to_param_names, py.name, py.namespace),
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
            *_joined_parts(params, py.types, dk.parameter_types),
        )
        .select(
            py.namespace,
            py.name,
            pl.col("param_names_list").list.len().pipe(_to_func, py, ParamLists(), dk),
        )
        .sort(py.namespace, py.name)
    )


def _filters(lf: pl.LazyFrame, dk: DuckCols) -> pl.LazyFrame:
    """First-step filter to remove unwanted functions."""
    return lf.select(dk.to_dict().keys()).filter(
        dk.function_type.is_in(
            {FuncTypes.TABLE, FuncTypes.TABLE_MACRO, FuncTypes.PRAGMA}
        ).not_(),
        dk.parameters.list.len().eq(0).and_(dk.varargs.is_null()).not_(),  # literals
        dk.function_name.is_in(SPECIAL_CASES).not_(),
        *PREFIXES.iter().map(
            lambda prefix: dk.function_name.str.starts_with(prefix).not_()
        ),
    )


def _py_name(dk: DuckCols, py: PyCols) -> pl.Expr:
    def _strip_namespace_prefixes(py_name: pl.Expr, namespace: pl.Expr) -> pl.Expr:
        def _strip_prefixes(expr: pl.Expr, prefixes: pc.Seq[str]) -> pl.Expr:
            return prefixes.iter().fold(
                expr,
                lambda acc, prefix: (
                    pl.when(acc.str.starts_with(prefix))
                    .then(acc.str.slice(len(prefix)))
                    .otherwise(acc)
                ),
            )

        return pl.coalesce(
            *NAMESPACE_SPECS.iter()
            .filter(lambda spec: spec.strip_prefixes.length() > 0)
            .map(
                lambda spec: pl.when(namespace.eq(spec.name)).then(
                    py_name.pipe(_strip_prefixes, spec.strip_prefixes)
                )
            ),
            py_name,
        )

    return (
        pl.concat_str(
            dk.function_name,
            pl.when(py.suffixes.is_not_null()).then(
                pl.concat_str(pl.lit("_"), py.suffixes)
            ),
            ignore_nulls=True,
        )
        .str.to_lowercase()
        .pipe(_strip_namespace_prefixes, py.namespace)
        .alias("py_name")
    )


def _alias_map(lf: pl.LazyFrame, dk: DuckCols) -> pl.LazyFrame:
    """Map of `function_name` to list of aliases.

    Alias root is determined by taking the first value between `function_name` and `alias_of` that is not null.
    Then all other `function_name`s that share the same `alias_root` are considered aliases of each other.
    """
    return (
        lf.select(
            dk.function_name,
            dk.alias_of,
            pl.coalesce(dk.alias_of, dk.function_name).alias("alias_root"),
        )
        .pipe(
            lambda lf: lf.join(
                lf.group_by("alias_root").agg(
                    dk.function_name.unique().sort().alias("alias_group")
                ),
                on="alias_root",
                how="left",
            )
        )
        .select(
            dk.function_name,
            pl.col("alias_group")
            .list.set_difference(pl.concat_list(dk.function_name))
            .alias("aliases"),
        )
        .unique()
    )


def _into_union(expr: pl.Expr) -> pl.Expr:
    return expr.filter(expr.ne(EMPTY_STR)).unique().sort().str.join(" | ")


def _joined_parts(
    params: Params, py_union: pl.Expr, params_union: pl.Expr
) -> Iterable[pl.Expr]:
    cond = params.idx.ge(params.lens.min_params_per_fn)

    return (
        params.names.alias("param_names_list"),
        format_kwords(
            "{param_name}: {py_type}{union}",
            param_name=params.names,
            py_type=py_union,
            union=pl.when(cond).then(pl.lit(" | None = None")),
            ignore_nulls=True,
        ).alias("param_sig_list"),
        format_kwords(
            "            {param_name} ({py_type}{union}): `{dk_type}` expression",
            param_name=params.names,
            py_type=py_union,
            union=pl.when(cond).then(pl.lit(" | None")),
            dk_type=params_union,
            ignore_nulls=True,
        ).alias("param_doc_list"),
    )


def _make_type_union(py_type: pl.Expr, self_type: pl.Expr) -> pl.Expr:
    txt = "{self_type} | {py_type}"

    return (
        pl.when(py_type.eq(EMPTY_STR))
        .then(self_type)
        .otherwise(format_kwords(txt, self_type=self_type, py_type=py_type))
    )


def _namespace_specs(cats: pl.Expr, fn_name: pl.Expr) -> pl.Expr:
    empty_lst = pl.lit(None, dtype=pl.List(pl.String))
    return (
        pl.when(fn_name.is_in(GENERIC_FUNCTIONS))
        .then(empty_lst)
        .otherwise(
            NAMESPACE_SPECS.iter()
            .map(
                lambda spec: pl.when(
                    spec.categories.iter()
                    .map(lambda c: c.value)
                    .into(lambda x: cats.list.set_intersection(x.collect(tuple)))
                    .list.len()
                    .gt(0)
                ).then(pl.lit(spec.name))
            )
            .chain(
                NAMESPACE_SPECS.iter().map(
                    lambda spec: pl.when(
                        spec.prefixes.iter()
                        .map(fn_name.str.starts_with)
                        .into(pl.any_horizontal)
                    ).then(pl.lit(spec.name))
                )
            )
            .into(pl.concat_list)
            .list.drop_nulls()
            .list.unique()
            .pipe(
                lambda expr: (
                    pl.when(expr.list.len().gt(0)).then(expr).otherwise(empty_lst)
                )
            )
        )
        .alias("namespace")
    )


def _convert_duckdb_type_to_python(param_type: pl.Expr) -> pl.Expr:
    return param_type.replace_strict(CONVERTER, return_dtype=pl.String)


def _to_param_names(params: pl.Expr, py_name: pl.Expr, namespace: pl.Expr) -> pl.Expr:
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
                pl.when(expr.cum_count().over(py_name, namespace, expr).gt(1))
                .then(
                    format_kwords(
                        "{name}_{count}",
                        name=expr,
                        count=expr.cum_count()
                        .over(py_name, namespace, expr)
                        .cast(pl.String),
                    )
                )
                .otherwise(expr)
            )
        )
        .alias("param_names")
    )


def _py_name_map(lf: pl.LazyFrame, dk: DuckCols, p_lens: ParamLens) -> pl.LazyFrame:
    return (
        lf.filter(
            dk.description.n_unique()
            .over(dk.function_name, dk.categories)
            .gt(1)
            .and_(p_lens.min_params_per_fn_cat_desc.gt(p_lens.min_params_per_fn))
            .and_(p_lens.sig_param_count.eq(p_lens.min_params_per_fn_cat_desc))
        )
        .select(
            dk.function_name,
            dk.categories,
            dk.description,
            pl.when(p_lens.min_params_per_fn_cat_desc.eq(p_lens.min_params_per_fn))
            .then(dk.parameters)
            .otherwise(
                dk.parameters.list.slice(
                    p_lens.min_params_per_fn,
                    p_lens.sig_param_count.sub(p_lens.min_params_per_fn),
                )
            )
            .list.join("_")
            .alias("py_suffixes"),
        )
        .unique()
    )


def _to_func(
    has_params: pl.Expr, py: PyCols, p_lists: ParamLists, dk: DuckCols
) -> pl.Expr:

    varargs_type = dk.varargs.pipe(_convert_duckdb_type_to_python).pipe(
        _make_type_union, py.self_type
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
        see_also_section=pl.when(py.aliases.list.len().gt(0)).then(
            format_kwords(
                "\n\n        See Also:\n            {aliases}",
                aliases=py.aliases.list.sort().list.join(", "),
                ignore_nulls=True,
            )
        ),
        args_section=pl.when(has_params.gt(1).or_(dk.varargs.is_not_null())).then(
            format_kwords(
                "\n\n        Args:\n{posargs}{varargs}",
                posargs=p_lists.docs.list.slice(1).list.join("\n"),
                varargs=pl.when(dk.varargs.is_not_null()).then(
                    format_kwords(
                        "\n            *args ({pytypes}): `{dk_types}` expression",
                        pytypes=varargs_type,
                        dk_types=dk.varargs,
                    )
                ),
                ignore_nulls=True,
            )
        ),
        sql_name=dk.function_name,
        dk_name=dk.function_name,
        dk_args=pl.when(has_params.gt(1)).then(
            format_kwords(", {args}", args=p_lists.names.list.slice(1).list.join(", "))
        ),
        dk_varargs=pl.when(dk.varargs.is_not_null()).then(pl.lit(", *args")),
        ignore_nulls=True,
    )


def _txt() -> str:
    return '''
    def {func_name}(self{args}{varargs}) -> {self_type}:
        """{description}.

        **SQL name**: *{sql_name}*{see_also_section}{args_section}

        Returns:
            {self_type}
        """
        return self._new(func("{dk_name}", self.inner(){dk_args}{dk_varargs}))
        '''
