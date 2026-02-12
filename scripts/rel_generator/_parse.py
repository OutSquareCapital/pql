import ast
import inspect
from pathlib import Path
from typing import TypeIs

import duckdb
import pyochain as pc

from ._structs import MethodInfo, ParamInfo, PyLit


def extract_methods_from_stub(stub_path: Path) -> pc.Seq[MethodInfo]:
    """Parse the .pyi stub and extract all DuckDBPyRelation methods."""
    docs = _get_runtime_docs()

    def _is_relation_class(node: ast.AST) -> TypeIs[ast.ClassDef]:
        return isinstance(node, ast.ClassDef) and node.name == PyLit.DUCK_REL.value

    return (
        pc.Iter(ast.iter_child_nodes(ast.parse(stub_path.read_text(encoding="utf-8"))))
        .find(_is_relation_class)
        .map(
            lambda cls: (
                pc.Iter(cls.body)  # type: ignore[typeis]
                .filter_map(
                    lambda item: _parse_method(item, docs)  # type: ignore[typeis]
                )
                .collect()
            )
        )
        .expect("DuckDBPyRelation class not found in stub")
    )


def _parse_method(node: ast.stmt, docs: pc.Dict[str, str]) -> pc.Option[MethodInfo]:
    """Parse a single ``ast.FunctionDef`` into ``MethodInfo``."""
    if not isinstance(node, ast.FunctionDef):
        return pc.NONE

    def _is_decorator(d: ast.expr, target: str) -> bool:
        match d:
            case ast.Attribute(attr=attr):
                return attr == target
            case ast.Name(id=id_):
                return id_ == target
            case _:
                return False

    regular_args = pc.Vec.from_ref(node.args.args[1:])
    defaults = pc.Vec.from_ref(node.args.defaults)
    num_no_default = regular_args.length() - defaults.length()

    params = (
        regular_args.iter()
        .enumerate()
        .map_star(
            lambda i, arg: ParamInfo(
                name=arg.arg,
                annotation=ast.unparse(arg.annotation)
                if arg.annotation
                else PyLit.ANY.value,
                default=(
                    pc.NONE
                    if i < num_no_default
                    else pc.Some(ast.unparse(defaults[i - num_no_default]))
                ),
            )
        )
        .chain(
            pc.Iter(node.args.kwonlyargs)
            .zip(node.args.kw_defaults, strict=False)
            .map_star(
                lambda arg, default: ParamInfo(
                    name=arg.arg,
                    annotation=ast.unparse(arg.annotation)
                    if arg.annotation
                    else PyLit.ANY.value,
                    default=pc.Option(default).map(ast.unparse),
                    is_kw_only=True,
                )
            )
        )
    )

    return pc.Some(
        MethodInfo(
            name=node.name,
            params=params.collect(pc.Vec),
            vararg=pc.Option(node.args.vararg).map(
                lambda v: ParamInfo(
                    name=v.arg,
                    annotation=ast.unparse(v.annotation) if v.annotation else "Any",
                )
            ),
            return_type=ast.unparse(node.returns) if node.returns else PyLit.NONE.value,
            is_overload=pc.Iter(node.decorator_list).any(
                lambda d: _is_decorator(d, "overload")
            ),
            is_property=pc.Iter(node.decorator_list).any(
                lambda d: _is_decorator(d, "property")
            ),
            doc=docs.get_item(node.name).unwrap_or(""),
        )
    )


def _get_runtime_docs() -> pc.Dict[str, str]:
    """Get docstrings from the runtime DuckDBPyRelation class."""
    return (
        pc.Iter(inspect.getmembers(duckdb.DuckDBPyRelation))
        .filter_map_star(
            lambda name, member: (
                pc.Option(inspect.getdoc(member))
                .map(lambda doc: _strip_sig_from_doc(name, doc))
                .map(lambda doc: (name, doc))
            )
        )
        .collect(pc.Dict)
    )


def _strip_sig_from_doc(name: str, doc: str) -> str:
    """Strip the signature line from the docstring (duckdb embeds it)."""
    return (
        pc.Vec.from_ref(doc.splitlines())
        .then_some()
        .filter(lambda lines: lines.first().strip().startswith(f"{name}("))
        .map(
            lambda lines: (
                lines.iter().skip(1).skip_while(lambda line: line == "").join("\n")
            )
        )
        .unwrap_or(doc)
    )
