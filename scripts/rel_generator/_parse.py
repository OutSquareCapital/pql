import ast
import inspect
from pathlib import Path
from typing import TypeIs

import duckdb
import pyochain as pc

from ._rules import fix_kw_only, fix_rel_param, fix_rel_return
from ._structs import MethodInfo, ParamInfo, PyLit


def extract_methods_from_stub(stub_path: Path) -> pc.Seq[MethodInfo]:
    """Parse the .pyi stub and extract all DuckDBPyRelation methods."""
    docs = _get_runtime_docs()

    def _is_relation_class(node: ast.AST) -> TypeIs[ast.ClassDef]:
        return isinstance(node, ast.ClassDef) and node.name == PyLit.DUCK_REL

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
        .expect(f"{PyLit.DUCK_REL} class not found in stub")
    )


def _parse_method(node: ast.stmt, docs: pc.Dict[str, str]) -> pc.Option[MethodInfo]:
    """Parse a single ``ast.FunctionDef`` into ``MethodInfo``."""
    match node:
        case ast.FunctionDef():
            regular_args = pc.Iter(node.args.args).skip(1).collect()
            defaults = pc.Vec.from_ref(node.args.defaults)
            num_no_default = regular_args.length() - defaults.length()

            def _check_decorator(target: PyLit) -> bool:
                return pc.Iter(node.decorator_list).any(
                    lambda d: _is_decorator(d, target)
                )

            return pc.Some(
                MethodInfo(
                    name=node.name,
                    params=regular_args.into(
                        _get_params, defaults, node, num_no_default
                    ),
                    vararg=pc.Option(node.args.vararg).map(
                        lambda v: _to_param(node, v.arg, pc.Option(v.annotation))
                    ),
                    return_type=fix_rel_return(
                        node.name,
                        pc.Option(node.returns).map(ast.unparse).unwrap_or(PyLit.NONE),
                    ),
                    is_overload=_check_decorator(PyLit.OVERLOAD),
                    is_property=_check_decorator(PyLit.PROPERTY),
                    doc=docs.get_item(node.name).unwrap_or(""),
                )
            )

        case _:
            return pc.NONE


def _get_params(
    regular_args: pc.Seq[ast.arg],
    defaults: pc.Vec[ast.expr],
    node: ast.FunctionDef,
    num_no_default: int,
) -> pc.Seq[ParamInfo]:
    return (
        regular_args.iter()
        .enumerate()
        .map_star(
            lambda i, arg: _to_param(
                node,
                arg.arg,
                pc.Option(arg.annotation),
                (
                    pc.NONE
                    if i < num_no_default
                    else pc.Some(ast.unparse(defaults[i - num_no_default]))
                ),
                is_kw_only=fix_kw_only(node.name) and i > 0,
            )
        )
        .chain(
            pc.Iter(node.args.kwonlyargs)
            .zip(node.args.kw_defaults, strict=False)
            .map_star(
                lambda arg, default: _to_param(
                    node,
                    arg.arg,
                    pc.Option(arg.annotation),
                    pc.Option(default).map(ast.unparse),
                    is_kw_only=True,
                )
            )
        )
        .collect()
    )


def _to_param(
    node: ast.FunctionDef,
    name: str,
    annotation: pc.Option[ast.expr],
    default: pc.Option[str] = pc.NONE,
    *,
    is_kw_only: bool = False,
) -> ParamInfo:
    return ParamInfo(
        name,
        fix_rel_param(
            node.name,
            name,
            annotation.map(ast.unparse).unwrap_or(PyLit.ANY),
        ),
        default,
        is_kw_only,
    )


def _is_decorator(d: ast.expr, target: str) -> bool:
    match d:
        case ast.Attribute(attr=attr):
            return attr == target
        case ast.Name(id=id_):
            return id_ == target
        case _:
            return False


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
