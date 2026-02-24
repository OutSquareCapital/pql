from __future__ import annotations

import ast
from dataclasses import dataclass, field
from textwrap import indent
from typing import NamedTuple, Self

import pyochain as pc
from astToolkit import Make

from .._utils import Builtins, CollectionsABC, DuckDB, Pql, Pyochain, Typing
from ._target import REL_TARGET, ReturnMeta, TargetSpec

ARG = "arg"
INNER = "inner"


@dataclass(slots=True)
class ParamInfo:
    """A single parameter extracted from the stub."""

    name: str
    annotation: str
    default: pc.Option[str] = field(default_factory=lambda: pc.NONE)
    is_kw_only: bool = False

    def forward_vararg(self, target: TargetSpec) -> ast.Starred:
        """Generate the forwarded vararg, converting types at boundary."""
        match target:
            case t if t == REL_TARGET:
                match target.rewrite_type(self.annotation):
                    case a if Pql.DUCK_HANDLER in a and Builtins.STR not in a:
                        expr = _iter_map_inner(_iter_from_name(self.name))
                    case a if Pql.DUCK_HANDLER in a:
                        expr = _iter_map_name(
                            _iter_from_name(self.name), Pql.INTO_DUCKDB
                        )
                    case _:
                        expr = Make.Name(self.name)
            case _ if target.stub_class in self.annotation:
                expr = _iter_map_inner(_iter_from_name(self.name))
            case _:
                expr = Make.Name(self.name)
        return ast.Starred(value=expr, ctx=ast.Load())

    def forward_arg(self, target: TargetSpec) -> ast.expr:  # noqa: PLR0911
        """Generate the forwarded argument, converting types at boundary."""
        match target:
            case t if t == REL_TARGET:
                match self.annotation:
                    case a if CollectionsABC.ITERABLE in a and DuckDB.EXPRESSION in a:
                        return _iter_map_name(
                            _call_name(Pql.TRY_ITER, Make.Name(self.name)),
                            Pql.INTO_DUCKDB,
                        )
                    case a if DuckDB.EXPRESSION in a and DuckDB.RELATION not in a:
                        return _call_name(Pql.INTO_DUCKDB, Make.Name(self.name))
                    case a if DuckDB.RELATION in a:
                        return _call_attr0(Make.Name(self.name), INNER)
                    case _:
                        return Make.Name(self.name)
            case _:
                match self.annotation:
                    case a if CollectionsABC.ITERABLE in a and target.stub_class in a:
                        return _iter_map_inner(
                            _call_name(Pql.TRY_ITER, Make.Name(self.name))
                        )
                    case a if target.stub_class in a or DuckDB.RELATION in a:
                        return _call_attr0(Make.Name(self.name), INNER)
                    case _:
                        return Make.Name(self.name)


@dataclass(slots=True)
class MethodInfo:
    """A method extracted from the stub + runtime doc."""

    name: str
    params: pc.Seq[ParamInfo]
    vararg: pc.Option[ParamInfo]
    return_type: str
    is_overload: bool
    is_property: bool
    doc: str
    target: TargetSpec

    def as_impl(self) -> Self:
        self.is_overload = False
        return self

    @property
    def returns_relation(self) -> bool:
        return self.return_type == self.target.wrapped_return_type

    @property
    def returns_none(self) -> bool:
        return self.return_type == Builtins.NONE

    def _return_meta(self) -> ReturnMeta:
        if self.returns_relation:
            return ReturnMeta(Typing.SELF, pc.NONE)
        return self.target.return_meta(self.return_type)

    def generate_method(self) -> str:
        """Generate a single method wrapper."""
        return_annotation, wrapper = self._return_meta()
        ast_signature = self._build_ast_signature(return_annotation)
        match self.name:
            case _ if self.is_overload:
                return ast_signature.render_overload()
            case _ if self.is_property:
                return self._build_property_signature(return_annotation).render(
                    self.doc,
                    _wrap_return_expr(
                        _call_attr(
                            _call_attr0(Make.Name(Builtins.SELF), INNER), self.name
                        ),
                        wrapper,
                    ),
                    decorators=pc.Seq((Builtins.PROPERTY,)),
                )
            case _:
                return ast_signature.render(self.doc, self._build_return_expr(wrapper))

    def _build_ast_signature(self, return_annotation: str) -> AstSignature:
        pos_params = self.params.iter().filter(lambda p: not p.is_kw_only).collect()
        kwonly_params = self.params.iter().filter(lambda p: p.is_kw_only).collect()
        args = Make.arguments(
            list_arg=pos_params.iter()
            .map(lambda p: _make_arg(p.name, self.target.rewrite_type(p.annotation)))
            .insert(Make.arg(Builtins.SELF))
            .collect(list),
            vararg=self.vararg.map(
                lambda p: _make_arg(p.name, self.target.rewrite_type(p.annotation))
            ).unwrap_or(None),  # pyright: ignore[reportArgumentType]
            kwonlyargs=kwonly_params.iter()
            .map(lambda p: _make_arg(p.name, self.target.rewrite_type(p.annotation)))
            .collect(list),
            kw_defaults=kwonly_params.iter()
            .map(
                lambda p: p.default.map(_to_expr).unwrap_or(None)  # pyright: ignore[reportArgumentType]
            )
            .collect(),
            defaults=(
                pos_params.iter()
                .map(lambda p: p.default)
                .collect(pc.Vec)
                .into(
                    lambda pos_defaults: (
                        pos_defaults.iter()
                        .enumerate()
                        .find_map(
                            lambda item: (
                                pc.Some(item[0]) if item[1].is_some() else pc.NONE
                            )
                        )
                        .map(
                            lambda idx: (
                                pos_defaults.iter()
                                .skip(idx)
                                .map(
                                    lambda d: d.expect(
                                        "Expected trailing default value"
                                    )
                                )
                                .map(_to_expr)
                                .collect()
                            )
                        )
                        .unwrap_or(pc.Seq(()))
                    )
                )
            ),
        )

        return AstSignature(
            self.target.rename_method(self.name), args, _to_expr(return_annotation)
        )

    def _build_property_signature(self, return_annotation: str) -> AstSignature:
        return AstSignature(
            self.target.rename_method(self.name),
            Make.arguments(list_arg=[Make.arg(Builtins.SELF)]),
            _to_expr(return_annotation),
        )

    def _build_return_expr(self, wrapper: pc.Option[str]) -> ast.expr:
        args = (
            self.params.iter()
            .filter(lambda p: not p.is_kw_only)
            .map(lambda p: p.forward_arg(self.target))
        )
        kwargs = (
            self.params.iter()
            .filter(lambda p: p.is_kw_only)
            .map(lambda p: ast.keyword(arg=p.name, value=p.forward_arg(self.target)))
        )

        call = ast.Call(
            func=_call_attr(_call_attr0(Make.Name(Builtins.SELF), INNER), self.name),
            args=self.vararg.map_or(
                args.collect(list),
                lambda v: args.insert(v.forward_vararg(self.target)).collect(list),
            ),
            keywords=kwargs.collect(list),
        )

        if self.returns_relation:
            return _call_attr(Make.Name(Builtins.SELF), "_new", call)
        return _wrap_return_expr(call, wrapper)


class AstSignature(NamedTuple):
    name: str
    args: ast.arguments
    returns: ast.expr

    def render(
        self,
        doc: str,
        return_expr: ast.expr,
        decorators: pc.Seq[str] | None = None,
    ) -> str:
        body = pc.Option.if_some(doc.strip()).map_or(
            [Make.Return(return_expr)],
            lambda txt: [
                Make.Expr(Make.Constant(txt)),
                Make.Return(return_expr),
            ],
        )

        rendered = Make.FunctionDef(
            name=self.name,
            argumentSpecification=self.args,
            body=body,
            decorator_list=pc.Option(decorators)
            .map(lambda ds: ds.iter().map(Make.Name).collect())
            .unwrap_or(pc.Seq(())),
            returns=self.returns,
        )
        ast.fix_missing_locations(rendered)
        return indent(ast.unparse(rendered), "    ")

    def render_overload(self) -> str:
        rendered = Make.FunctionDef(
            name=self.name,
            argumentSpecification=self.args,
            body=[Make.Expr(Make.Constant(Ellipsis))],
            decorator_list=[Make.Name(Typing.OVERLOAD)],
            returns=self.returns,
        )
        ast.fix_missing_locations(rendered)
        return indent(ast.unparse(rendered), "    ")


def _to_expr(expr: str) -> ast.expr:
    return ast.parse(expr, mode="eval").body


def _make_arg(name: str, annotation: str) -> ast.arg:
    return Make.arg(name, _to_expr(annotation))


def _wrap_return_expr(value: ast.expr, wrapper: pc.Option[str]) -> ast.expr:
    return wrapper.map_or(
        value,
        lambda fn_name: ast.Call(func=_to_expr(fn_name), args=[value]),
    )


def _iter_map_inner(iterable_expr: ast.expr) -> ast.expr:
    lambda_expr = ast.Lambda(
        args=ast.arguments(args=[Make.arg(ARG)]),
        body=_call_attr0(Make.Name(ARG), INNER),
    )
    return _call_attr(iterable_expr, "map", lambda_expr)


def _iter_from_name(name: str) -> ast.expr:
    return _call_attr(Make.Name("pc"), Pyochain.ITER, Make.Name(name))


def _call_name(name: str, *args: ast.expr) -> ast.Call:
    return ast.Call(func=Make.Name(name), args=list(args))


def _iter_map_name(iterable_expr: ast.expr, fn_name: str) -> ast.expr:
    return _call_attr(iterable_expr, "map", Make.Name(fn_name))


def _call_attr(obj: ast.expr, attr: str, *args: ast.expr) -> ast.expr:
    node = ast.Attribute(value=obj, attr=attr, ctx=ast.Load())
    return node if not args else ast.Call(func=node, args=list(args))


def _call_attr0(obj: ast.expr, attr: str) -> ast.Call:
    return ast.Call(ast.Attribute(value=obj, attr=attr, ctx=ast.Load()))
