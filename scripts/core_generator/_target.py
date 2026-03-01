import inspect
from dataclasses import dataclass, field
from typing import NamedTuple

import duckdb
import pyochain as pc

from .._utils import Builtins, DuckDB, Dunders, Pql, Pyochain, Typing
from ._rules import PYTYPING_REWRITES


class ReturnMeta(NamedTuple):
    return_annotation: str
    wrapper: pc.Option[str]


@dataclass(slots=True)
class TargetSpec:
    stub_file_name: str
    stub_class: str
    wrapper_class: str
    wrapper_base: str
    wrapped_return_type: str
    skip_methods: pc.Set[str] = field(default_factory=pc.Set[str].new)
    method_renames: pc.Dict[str, str] = field(default_factory=pc.Dict[str, str].new)

    def rewrite_type(self, annotation: str) -> str:
        """Rewrite a stub type annotation for use in the generated wrapper."""

        def _sub_one(acc: str, old_new: tuple[str, str]) -> str:
            return acc.replace(old_new[0], old_new[1])

        return (
            PYTYPING_REWRITES.items()
            .iter()
            .fold(annotation, _sub_one)
            .replace(self.stub_class, Typing.SELF)
        )

    def return_meta(self, annotation: str) -> ReturnMeta:
        def _collection_kind() -> pc.Option[Builtins]:
            match rewritten:
                case _ if rewritten.startswith(Builtins.LIST):
                    return pc.Some(Builtins.LIST)
                case _ if rewritten.startswith("lst"):
                    return pc.Some(Builtins.LIST)
                case _ if rewritten.startswith(Builtins.DICT):
                    return pc.Some(Builtins.DICT)
                case _:
                    return pc.NONE

        def _build(collection: Pyochain, suffix: str) -> ReturnMeta:
            return ReturnMeta(
                f"pc.{collection}{suffix}"
                if suffix
                else f"pc.{collection.of_type(Typing.ANY)}",
                pc.Some(f"pc.{collection}.from_ref"),
            )

        rewritten = self.rewrite_type(annotation)
        match _collection_kind():
            case pc.Some(Builtins.LIST):
                suffix = rewritten.removeprefix(Builtins.LIST).removeprefix("lst")
                return _build(Pyochain.VEC, suffix)
            case pc.Some(Builtins.DICT):
                return _build(Pyochain.DICT, rewritten.removeprefix(Builtins.DICT))
            case _:
                return ReturnMeta(rewritten, pc.NONE)

    def rename_method(self, method_name: str) -> str:
        return self.method_renames.get_item(method_name).unwrap_or(method_name)

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


class Targets:
    RELATION = TargetSpec(
        stub_file_name="__init__.pyi",
        stub_class=DuckDB.RELATION,
        wrapper_class=Pql.RELATION,
        wrapper_base=Pql.REL_HANDLER,
        wrapped_return_type=DuckDB.RELATION,
    )

    EXPRESSION = TargetSpec(
        stub_file_name="_expression.pyi",
        stub_class=DuckDB.EXPRESSION,
        wrapper_class=DuckDB.EXPRESSION,
        wrapper_base=Pql.DUCK_HANDLER,
        wrapped_return_type=DuckDB.EXPRESSION,
        skip_methods=pc.Set({Dunders.INIT, "when", "otherwise"}),
        method_renames=pc.Dict.from_kwargs(
            isnull="is_null", isin="is_in", isnotin="is_not_in", isnotnull="is_not_null"
        ),
    )

    @classmethod
    def into_iter(cls) -> pc.Iter[TargetSpec]:
        return pc.Iter((cls.RELATION, cls.EXPRESSION))
