import ast
from pathlib import Path
from typing import TypeIs

import pyochain as pc

from ._structs import MethodInfo, ParamInfo, PyLit
from ._target import TargetSpec


def extract_methods_from_stub(
    stub_path: Path, target: TargetSpec
) -> pc.Seq[MethodInfo]:
    """Parse the .pyi stub and extract methods for a target DuckDB class."""
    docs = target.get_runtime_docs()

    def _is_target_class(node: ast.AST) -> TypeIs[ast.ClassDef]:
        return isinstance(node, ast.ClassDef) and node.name == target.stub_class

    return (
        pc.Iter(ast.iter_child_nodes(ast.parse(stub_path.read_text(encoding="utf-8"))))
        .find(_is_target_class)
        .map(
            lambda cls: (
                pc.Iter(cls.body)  # type: ignore[typeis]
                .filter_map(
                    lambda item: _parse_method(item, docs, target)  # type: ignore[typeis]
                )
                .collect()
            )
        )
        .expect(f"{target.stub_class} class not found in stub")
    )


def _parse_method(
    node: ast.stmt,
    docs: pc.Dict[str, str],
    target: TargetSpec,
) -> pc.Option[MethodInfo]:
    """Parse a single ``ast.FunctionDef`` into ``MethodInfo``."""
    match node:
        case ast.FunctionDef(name=name) if target.skip_methods.contains(name):
            return pc.NONE
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
                        _get_params, defaults, node, num_no_default, target
                    ),
                    vararg=pc.Option(node.args.vararg).map(
                        lambda v: _to_param(node.name, v, target)
                    ),
                    return_type=target.fix_return(
                        node.name,
                        pc.Option(node.returns).map(ast.unparse).unwrap_or(PyLit.NONE),
                    ),
                    is_overload=_check_decorator(PyLit.OVERLOAD),
                    is_property=_check_decorator(PyLit.PROPERTY),
                    doc=docs.get_item(node.name).unwrap_or(""),
                    target=target,
                )
            )

        case _:
            return pc.NONE


def _get_params(
    regular_args: pc.Seq[ast.arg],
    defaults: pc.Vec[ast.expr],
    node: ast.FunctionDef,
    num_no_default: int,
    target: TargetSpec,
) -> pc.Seq[ParamInfo]:
    return (
        regular_args.iter()
        .enumerate()
        .map_star(
            lambda i, arg: _to_param(
                node.name,
                arg,
                target,
                (
                    pc.NONE
                    if i < num_no_default
                    else pc.Some(ast.unparse(defaults[i - num_no_default]))
                ),
                is_kw_only=target.fix_kw_only(node.name) and i > 0,
            )
        )
        .chain(
            pc.Iter(node.args.kwonlyargs)
            .zip(node.args.kw_defaults, strict=False)
            .map_star(
                lambda arg, default: _to_param(
                    node.name,
                    arg,
                    target,
                    pc.Option(default).map(ast.unparse),
                    is_kw_only=True,
                )
            )
        )
        .collect()
    )


def _to_param(
    method_name: str,
    arg: ast.arg,
    target: TargetSpec,
    default: pc.Option[str] = pc.NONE,
    *,
    is_kw_only: bool = False,
) -> ParamInfo:
    return ParamInfo(
        arg.arg,
        target.fix_param(
            method_name,
            arg.arg,
            pc.Option(arg.annotation).map(ast.unparse).unwrap_or(PyLit.ANY),
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
