from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import duckdb
import polars as pl
import pyochain as pc

from ._models import (
    CONVERSION_MAP,
    FUNC_TYPES,
    KWORDS,
    NAMESPACE_SPECS,
    OPERATOR_MAP,
    SHADOWERS,
    DuckDbTypes,
    FuncTypes,
    NamespaceSpec,
    PyTypes,
)

_EMPTY_STR = pl.lit("")


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


def get_df() -> pl.LazyFrame:
    py = PyCols()
    params = Params()
    dk = DuckCols()
    return (
        dk.query()
        .pl(lazy=True)
        .filter(
            dk.function_type.cast(FUNC_TYPES)
            .is_in({FuncTypes.TABLE, FuncTypes.TABLE_MACRO, FuncTypes.PRAGMA})
            .not_(),
            dk.parameters.list.len()
            .eq(0)
            .and_(dk.varargs.is_null())
            .not_(),  # literals
            dk.function_name.is_in(OPERATOR_MAP).not_(),
            dk.function_name.str.starts_with("__").not_(),
            dk.function_name.str.starts_with("current_").not_(),  # Utility fns
            dk.function_name.str.starts_with("has_").not_(),  # Utility fns
            dk.function_name.str.starts_with("pg_").not_(),  # Postgres fns
            dk.function_name.str.starts_with("icu_").not_(),  # timestamp extension
            dk.function_name.ne("alias"),  # conflicts with duckdb alias method
            dk.alias_of.is_null().or_(dk.alias_of.is_in(OPERATOR_MAP)),
        )
        .with_columns(
            dk.parameter_types.list.eval(
                pl.element().fill_null(DuckDbTypes.ANY.value.upper())
            ),
            *dk.parameters.list.len().pipe(
                lambda expr_len: (
                    expr_len.alias("p_len_by_fn"),
                    expr_len.min().over(dk.function_name).alias("p_len_by_fn_cat"),
                    expr_len.min()
                    .over(dk.function_name, dk.categories, dk.description)
                    .alias("p_len_by_fn_cat_desc"),
                )
            ),
        )
        .with_columns(_to_py_name(dk, params.lens))
        .with_row_index("sig_id")
        .explode("parameters", "parameter_types")
        .with_columns(
            pl.int_range(pl.len()).over("sig_id").alias("param_idx"),
            dk.parameter_types.pipe(_convert_duckdb_type_to_python).alias("py_types"),
            dk.parameters.pipe(_to_param_names, py.name),
        )
        .group_by(py.name, params.idx, maintain_order=True)
        .agg(
            pl.all().exclude("py_types", "parameter_types").drop_nulls().first(),
            dk.parameter_types.pipe(_into_union),
            py.types.pipe(_into_union).pipe(_make_type_union),
        )
        .group_by(py.name, maintain_order=True)
        .agg(
            pl.all().exclude("param_names").first(),
            *params.names.pipe(
                lambda expr: expr.filter(expr.is_not_null()).pipe(
                    _joined_parts,
                    params.idx.ge(params.lens.by_fn_cat),
                    py.types,
                    dk.parameter_types,
                    _self_type(py.name, dk.categories),
                )
            ),
        )
        .select(
            py.name.pipe(_namespace_name, dk.categories).alias("namespace"),
            py.name,
            pl.col("param_names_list")
            .list.len()
            .pipe(
                _to_func,
                py.name,
                ParamLists(),
                dk.varargs.pipe(_convert_duckdb_type_to_python).pipe(_make_type_union),
                dk,
                py.name.pipe(_self_type, dk.categories),
                py.name.pipe(_self_expr, dk.categories),
                py.name.pipe(_return_ctor, dk.categories),
            ),
        )
        .sort("namespace", py.name)
    )


def _into_union(expr: pl.Expr) -> pl.Expr:
    return expr.unique().sort().str.join(" | ")


def _joined_parts(
    expr: pl.Expr,
    cond: pl.Expr,
    py_union: pl.Expr,
    params_union: pl.Expr,
    self_type: pl.Expr,
) -> Iterable[pl.Expr]:
    py_type = _replace_self(py_union, self_type)
    params_type = _replace_self(params_union, self_type)

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
            params_type,
            pl.lit("` expression"),
        ).str.join("\n")

    return (
        expr.alias("param_names_list"),
        expr.str.join(", ").alias("param_names_join"),
        _param_sig_list().alias("param_sig_list"),
        _param_doc_join().alias("param_doc_join"),
    )


def _make_type_union(py_type: pl.Expr) -> pl.Expr:
    return (
        pl.when(py_type.eq(pl.lit(PyTypes.SELF.value)))
        .then(py_type)
        .otherwise(pl.concat_str(pl.lit(f"{PyTypes.SELF.value} | "), py_type))
    )


def _replace_self(expr: pl.Expr, self_type: pl.Expr) -> pl.Expr:
    return expr.str.replace_all("Self", self_type)


def _namespace_name(fn_name: pl.Expr, categories: pl.Expr) -> pl.Expr:
    def _matches(spec_prefixes: pc.Seq[str]) -> pl.Expr:
        return spec_prefixes.iter().fold(
            pl.lit(value=False),
            lambda acc, prefix: acc.or_(fn_name.str.starts_with(prefix)),
        )

    def _matches_category(spec: NamespaceSpec) -> pl.Expr:
        return spec.categories.iter().fold(
            pl.lit(value=False),
            lambda acc, category: acc.or_(
                categories.list.contains(category).fill_null(value=False)
            ),
        )

    def _by_prefix() -> pl.Expr:
        return NAMESPACE_SPECS.iter().fold(
            pl.lit(value=None),
            lambda acc, spec: (
                pl.when(acc.is_not_null())
                .then(acc)
                .otherwise(
                    pl.when(_matches(spec.prefixes))
                    .then(pl.lit(spec.name))
                    .otherwise(acc)
                )
            ),
        )

    return (
        NAMESPACE_SPECS.iter()
        .fold(
            pl.lit(value=None),
            lambda acc, spec: (
                pl.when(acc.is_not_null())
                .then(acc)
                .otherwise(
                    pl.when(_matches_category(spec))
                    .then(pl.lit(spec.name))
                    .otherwise(acc)
                )
            ),
        )
        .pipe(
            lambda by_cat: (
                pl.when(by_cat.is_not_null()).then(by_cat).otherwise(_by_prefix())
            )
        )
    )


def _self_type(fn_name: pl.Expr, categories: pl.Expr) -> pl.Expr:
    return (
        pl.when(_namespace_name(fn_name, categories).is_not_null())
        .then(pl.lit("T"))
        .otherwise(pl.lit("Self"))
    )


def _self_expr(fn_name: pl.Expr, categories: pl.Expr) -> pl.Expr:
    return (
        pl.when(_namespace_name(fn_name, categories).is_not_null())
        .then(pl.lit("self._parent.inner()"))
        .otherwise(pl.lit("self._expr"))
    )


def _return_ctor(fn_name: pl.Expr, categories: pl.Expr) -> pl.Expr:
    return (
        pl.when(_namespace_name(fn_name, categories).is_not_null())
        .then(pl.lit("self._parent.__class__"))
        .otherwise(pl.lit("self.__class__"))
    )


def _convert_duckdb_type_to_python(param_type: pl.Expr) -> pl.Expr:
    converter = (
        CONVERSION_MAP.items()
        .iter()
        .map_star(
            lambda k, v: (
                k.value.upper(),
                v.iter().map(lambda pt: pt.value).join(" | "),
            )
        )
        .collect(dict)
    )

    def _replace(pattern: str) -> pl.Expr:
        return param_type.str.extract(pattern, 1).replace_strict(
            converter, default=PyTypes.SELF.value, return_dtype=pl.String
        )

    return (
        pl.when(
            param_type.str.contains(r"\[\]$").and_(
                param_type.is_in(("ANY[]", "T[]")).not_()
            )
        )
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
            lambda expr: (
                pl.when(expr.is_in(SHADOWERS))
                .then(pl.concat_str(expr, pl.lit("_arg")))
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


def _to_py_name(dk: DuckCols, p_lens: ParamLens) -> pl.Expr:
    return dk.categories.list.join("_").pipe(
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
                        .and_(p_lens.by_fn_cat_desc.gt(p_lens.by_fn_cat))
                    )
                    .then(
                        pl.concat_str(
                            base,
                            pl.lit("_"),
                            pl.when(p_lens.by_fn.eq(p_lens.by_fn_cat_desc))
                            .then(
                                pl.when(p_lens.by_fn_cat_desc.eq(p_lens.by_fn_cat))
                                .then(dk.parameters)
                                .otherwise(
                                    dk.parameters.list.slice(
                                        p_lens.by_fn_cat,
                                        p_lens.by_fn.sub(p_lens.by_fn_cat),
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


def _to_func(  # noqa: PLR0913
    has_params: pl.Expr,
    py_name: pl.Expr,
    p_lists: ParamLists,
    varargs_py_type: pl.Expr,
    dk: DuckCols,
    self_type: pl.Expr,
    self_expr: pl.Expr,
    return_ctor: pl.Expr,
) -> pl.Expr:
    varargs_type = _replace_self(varargs_py_type, self_type)

    def _signature(has_params: pl.Expr) -> pl.Expr:
        return pl.concat_str(
            pl.lit("    def "),
            py_name,
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
        return_ctor,
        pl.lit('(func("'),
        dk.function_name,
        pl.lit('"'),
        _body(has_params),
        pl.lit("))"),
    )
