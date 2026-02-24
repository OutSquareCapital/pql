import inspect
from dataclasses import dataclass, field
from textwrap import dedent
from typing import NamedTuple

import duckdb
import pyochain as pc

from .._utils import Builtins, DuckDB, Dunders, Pql, Pyochain, Typing
from ._rules import (
    EXPR_TYPE_SUBS,
    KW_ONLY_FIXES,
    PARAM_TYPE_FIXES,
    PYTYPING_REWRITES,
    RETURN_TYPE_FIXES,
    TYPE_SUBS,
)


class ReturnMeta(NamedTuple):
    return_annotation: str
    wrapper: pc.Option[str]


@dataclass(slots=True)
class TargetSpec:
    stub_class: str
    wrapper_class: str
    wrapper_base: str
    type_subs: pc.Dict[DuckDB, Pql | Typing]
    wrapped_return_type: str
    param_type_fixes: pc.Dict[str, pc.Dict[str, str]] = field(
        default_factory=pc.Dict[str, pc.Dict[str, str]].new
    )
    return_type_fixes: pc.Dict[str, str] = field(default_factory=pc.Dict[str, str].new)
    kw_only_fixes: pc.Set[str] = field(default_factory=pc.Set[str].new)
    skip_methods: pc.Set[str] = field(default_factory=pc.Set[str].new)
    method_renames: pc.Dict[str, str] = field(default_factory=pc.Dict[str, str].new)

    def rewrite_type(self, annotation: str) -> str:
        """Rewrite a stub type annotation for use in the generated wrapper."""

        def _sub_one(acc: str, old_new: tuple[str, str]) -> str:
            return acc.replace(old_new[0], old_new[1])

        return (
            self.type_subs.items()
            .iter()
            .fold(PYTYPING_REWRITES.items().iter().fold(annotation, _sub_one), _sub_one)
        )

    def return_meta(self, annotation: str) -> ReturnMeta:
        def _collection_kind() -> pc.Option[Builtins]:
            match rewritten:
                case _ if rewritten.startswith(Builtins.LIST):
                    return pc.Some(Builtins.LIST)
                case _ if rewritten.startswith(Builtins.DICT):
                    return pc.Some(Builtins.DICT)
                case _:
                    return pc.NONE

        def _build(collection: Pyochain, suffix: str) -> ReturnMeta:
            return ReturnMeta(
                f"pc.{collection}{suffix}"
                if suffix
                else f"pc.{collection}[{Typing.ANY}]",
                pc.Some(f"pc.{collection}.from_ref"),
            )

        rewritten = self.rewrite_type(annotation)
        match _collection_kind():
            case pc.Some(Builtins.LIST):
                return _build(Pyochain.VEC, rewritten.removeprefix(Builtins.LIST))
            case pc.Some(Builtins.DICT):
                return _build(Pyochain.DICT, rewritten.removeprefix(Builtins.DICT))
            case _:
                return ReturnMeta(rewritten, pc.NONE)

    def fix_param(self, method_name: str, param_name: str, annotation: str) -> str:
        return (
            self.param_type_fixes.get_item(method_name)
            .and_then(lambda params: params.get_item(param_name))
            .unwrap_or(annotation)
        )

    def fix_return(self, method_name: str, return_type: str) -> str:
        return self.return_type_fixes.get_item(method_name).unwrap_or(return_type)

    def fix_kw_only(self, method_name: str) -> bool:
        return self.kw_only_fixes.contains(method_name)

    def rename_method(self, method_name: str) -> str:
        return self.method_renames.get_item(method_name).unwrap_or(method_name)

    def class_def(self) -> str:
        return dedent(f'''

class {self.wrapper_class}({self.wrapper_base}):
    """Wrapper around {self.stub_class} that uses SqlExpr instead of duckdb.Expression.

    This is a composition-based wrapper: it stores a ``_expr: {self.stub_class}``
    and delegates all method calls, converting SqlExpr <-> duckdb.Expression
    at the boundary.
    """

    __slots__ = ()

    ''')

    def get_runtime_docs(self) -> pc.Dict[str, str]:
        """Get docstrings from the runtime target DuckDB class."""

        def _strip_sig_from_doc(name: str, doc: str) -> str:
            """Strip the signature line from the docstring (duckdb embeds it)."""
            return (
                pc.Vec.from_ref(doc.splitlines())
                .then_some()
                .filter(lambda lines: lines.first().strip().startswith(f"{name}("))
                .map(
                    lambda lines: (
                        lines.iter()
                        .skip(1)
                        .skip_while(lambda line: line == "")
                        .join("\n")
                    )
                )
                .unwrap_or(doc)
            )

        cls = getattr(duckdb, self.stub_class)

        return (
            pc.Iter(inspect.getmembers(cls))
            .filter_map_star(
                lambda name, member: (
                    pc.Option(inspect.getdoc(member))
                    .map(lambda doc: _strip_sig_from_doc(name, doc))
                    .map(lambda doc: (name, doc))
                )
            )
            .collect(pc.Dict)
        )


REL_TARGET = TargetSpec(
    stub_class=DuckDB.RELATION,
    wrapper_class=Pql.RELATION,
    wrapper_base=Pql.REL_HANDLER,
    type_subs=TYPE_SUBS,
    wrapped_return_type=DuckDB.RELATION,
    param_type_fixes=PARAM_TYPE_FIXES,
    return_type_fixes=RETURN_TYPE_FIXES,
    kw_only_fixes=KW_ONLY_FIXES,
)

EXPR_TARGET = TargetSpec(
    stub_class=DuckDB.EXPRESSION,
    wrapper_class=DuckDB.EXPRESSION,
    wrapper_base=Pql.DUCK_HANDLER,
    type_subs=EXPR_TYPE_SUBS,
    wrapped_return_type=DuckDB.EXPRESSION,
    skip_methods=pc.Set({Dunders.INIT, "when", "otherwise"}),
    method_renames=pc.Dict.from_kwargs(
        isnull="is_null", isin="is_in", isnotin="is_not_in", isnotnull="is_not_null"
    ),
)
