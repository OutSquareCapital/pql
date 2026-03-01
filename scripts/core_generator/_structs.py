from __future__ import annotations

import ast
from dataclasses import dataclass, field
from textwrap import indent
from typing import NamedTuple, Self

import pyochain as pc

from .._utils import Builtins, CollectionsABC, DuckDB, Pql, Pyochain, Typing
from ._rules import dunder_operator_alias
from ._target import ReturnMeta, Targets, TargetSpec

ARG = "arg"
INNER = "inner"
NEW = "_new"
MAP = "map"
INDENT = "    "


@dataclass(slots=True)
class Node(pc.traits.Pipeable):
    node: ast.expr

    @classmethod
    def ref(cls, name: str) -> Self:
        return cls(ast.Name(name))

    def attr(self, name: str) -> Self:
        return self.__class__(ast.Attribute(value=self.node, attr=name, ctx=ast.Load()))

    def call(self, *args: Self | ast.expr | str) -> Self:
        def _into_arg(x: Self | ast.expr | str) -> ast.expr:
            match x:
                case Node():
                    return x.node
                case str():
                    return self.ref(x).node
                case _:
                    return x

        return self.__class__(
            ast.Call(self.node, pc.Iter(args).map(_into_arg).collect(list))
        )

    def call_kw(self, args: list[ast.expr], keywords: list[ast.keyword]) -> Self:
        return self.__class__(ast.Call(func=self.node, args=args, keywords=keywords))

    def starred(self) -> ast.Starred:
        return ast.Starred(value=self.node, ctx=ast.Load())


def _map_inner(nb: Node) -> Node:
    lam = ast.Lambda(
        ast.arguments(args=[ast.arg(ARG)]), Node.ref(ARG).attr(INNER).call().node
    )
    return Node(ast.Call(func=nb.attr(MAP).node, args=[lam]))


def _wrap(nb: Node, wrapper: pc.Option[str]) -> Node:
    return wrapper.map_or(nb, lambda fn: Node.ref(fn).call(nb))


@dataclass(slots=True)
class ParamInfo:
    """A single parameter extracted from the stub."""

    name: str
    annotation: str
    default: pc.Option[str] = field(default_factory=lambda: pc.NONE)
    is_kw_only: bool = False

    def forward_vararg(self, target: TargetSpec) -> ast.Starred:
        """Return the `*args` AST node passed to the wrapped DuckDB call, mapping each item only when the rewritten annotation requires `into_col`, `into_duckdb`, or `.inner()`."""
        rewritten = target.rewrite_type(self.annotation)
        it = Node.ref("pc").attr(Pyochain.ITER).call(self.name)
        match target:
            case t if t == Targets.RELATION:
                match rewritten:
                    case a if Pql.INTO_EXPR in a or Pql.INTO_EXPR_COLUMN in a:
                        return it.attr(MAP).call(Pql.INTO_DUCKDB).starred()
                    case a if Pql.DUCK_HANDLER in a and Builtins.STR not in a:
                        return it.into(_map_inner).starred()
                    case a if Pql.DUCK_HANDLER in a:
                        return it.attr(MAP).call(Pql.INTO_DUCKDB).starred()
                    case _:
                        return Node.ref(self.name).starred()
            case _ if (
                Pql.INTO_EXPR in rewritten or target.stub_class in self.annotation
            ):
                return it.attr(MAP).call(Pql.INTO_DUCKDB).starred()
            case _:
                return Node.ref(self.name).starred()

    def forward_arg(self, target: TargetSpec) -> ast.expr:
        """Return the AST node for the argument passed to the wrapped DuckDB call, applying `into_col`, `into_duckdb`, iterable mapping, or `.inner()` only when required by the rewritten annotation."""
        rewritten = target.rewrite_type(self.annotation)
        name_e = Node.ref(self.name)
        if CollectionsABC.ITERABLE in rewritten and (
            Pql.INTO_EXPR in rewritten or Pql.INTO_EXPR_COLUMN in rewritten
        ):
            return (
                Node.ref(Pql.TRY_ITER)
                .call(self.name)
                .attr(MAP)
                .call(Pql.INTO_DUCKDB)
                .node
            )
        if CollectionsABC.MAPPING in rewritten and Pql.INTO_EXPR in rewritten:
            return Node.ref(Pql.INTO_DUCKDB_MAPPING).call(name_e).node
        match target:
            case t if t == Targets.RELATION:
                use_duckdb = (
                    Pql.INTO_EXPR in rewritten or Pql.INTO_EXPR_COLUMN in rewritten
                ) and DuckDB.RELATION not in rewritten
                use_inner = not use_duckdb and (
                    DuckDB.RELATION in self.annotation or Typing.SELF in rewritten
                )
            case _:
                use_duckdb = (
                    Pql.INTO_EXPR in rewritten or Pql.INTO_EXPR_COLUMN in rewritten
                )
                use_inner = not use_duckdb and (
                    target.stub_class in rewritten or DuckDB.RELATION in rewritten
                )
        if use_duckdb:
            return Node.ref(Pql.INTO_DUCKDB).call(name_e).node
        if use_inner:
            return name_e.attr(INNER).call().node
        return name_e.node


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

    def _meta(self) -> ReturnMeta:
        if self.returns_relation:
            return ReturnMeta(Typing.SELF, pc.NONE)
        return self.target.return_meta(self.return_type)

    def generate_method(self) -> str:
        """Generate a single method wrapper."""
        meta = self._meta()
        ast_signature = self._build_ast_signature(meta.return_annotation)
        match self.name:
            case _ if self.is_overload:
                return ast_signature.render_overload()
            case _ if self.is_property:
                return self._build_property_signature(meta.return_annotation).render(
                    self.doc,
                    Node.ref(Builtins.SELF)
                    .attr(INNER)
                    .call()
                    .attr(self.name)
                    .into(_wrap, meta.wrapper)
                    .node,
                    decorators=pc.Some(pc.Iter.once(Builtins.PROPERTY)),
                )
            case _:
                return ast_signature.render(
                    self.doc, self._build_return_expr(meta.wrapper)
                )

    def generate_methods(self) -> pc.Iter[str]:
        generated = self.generate_method()
        return self._build_operator_alias().map_or(
            pc.Iter.once(generated), lambda alias: pc.Iter((generated, alias))
        )

    def _build_operator_alias(self) -> pc.Option[str]:
        meta = self._meta()
        if self.target != Targets.EXPRESSION or self.is_overload or self.is_property:
            return pc.NONE
        return dunder_operator_alias(self.name).map(
            lambda alias_name: AstSignature(
                alias_name,
                self._build_ast_signature(meta.return_annotation).args,
                _to_expr(meta.return_annotation),
            ).render("", self._operator_alias_call())
        )

    def _operator_alias_call(self) -> ast.expr:
        pos_args = (
            self.params.iter()
            .filter(lambda p: not p.is_kw_only)
            .map(lambda p: Node.ref(p.name).node)
        )
        keywords = (
            self.params.iter()
            .filter(lambda p: p.is_kw_only)
            .map(lambda p: ast.keyword(arg=p.name, value=Node.ref(p.name).node))
            .collect(list)
        )
        call_args = self.vararg.map_or(
            pos_args.collect(list),
            lambda vararg: pos_args.insert(Node.ref(vararg.name).starred()).collect(
                list
            ),
        )
        return Node.ref(Builtins.SELF).attr(self.name).call_kw(call_args, keywords).node

    def _build_ast_signature(self, return_annotation: str) -> AstSignature:
        def _param_arg(p: ParamInfo) -> ast.arg:
            return _make_arg(p.name, self.target.rewrite_type(p.annotation))

        pos_params = self.params.iter().filter(lambda p: not p.is_kw_only).collect()
        kwonly_params = self.params.iter().filter(lambda p: p.is_kw_only).collect()
        return AstSignature(
            self.target.rename_method(self.name),
            ast.arguments(
                args=pos_params.iter()
                .map(_param_arg)
                .insert(ast.arg(Builtins.SELF))
                .collect(list),
                vararg=self.vararg.map(_param_arg).unwrap_or(None),  # pyright: ignore[reportArgumentType]
                kwonlyargs=kwonly_params.iter().map(_param_arg).collect(list),
                kw_defaults=kwonly_params.iter()
                .map(lambda p: p.default.map(_to_expr).unwrap_or(None))  # pyright: ignore[reportArgumentType]
                .collect(),
                defaults=pos_params.iter()
                .map(lambda p: p.default)
                .collect(pc.Vec)
                .into(
                    lambda ds: (
                        ds.iter()
                        .enumerate()
                        .find_map(
                            lambda item: (
                                pc.Some(item[0]) if item[1].is_some() else pc.NONE
                            )
                        )
                        .map(
                            lambda idx: (
                                ds.iter()
                                .skip(idx)
                                .map(
                                    lambda d: d.expect(
                                        "Expected trailing default value"
                                    )
                                )
                                .map(_to_expr)
                                .collect(list)
                            )
                        )
                        .unwrap_or([])
                    )
                ),
            ),
            _to_expr(return_annotation),
        )

    def _build_property_signature(self, return_annotation: str) -> AstSignature:
        return AstSignature(
            self.target.rename_method(self.name),
            ast.arguments(args=[ast.arg(Builtins.SELF)]),
            _to_expr(return_annotation),
        )

    def _build_return_expr(self, wrapper: pc.Option[str]) -> ast.expr:
        pos_args = (
            self.params.iter()
            .filter(lambda p: not p.is_kw_only)
            .map(lambda p: p.forward_arg(self.target))
        )
        keywords = (
            self.params.iter()
            .filter(lambda p: p.is_kw_only)
            .map(lambda p: ast.keyword(arg=p.name, value=p.forward_arg(self.target)))
            .collect(list)
        )
        call = (
            Node.ref(Builtins.SELF)
            .attr(INNER)
            .call()
            .attr(self.name)
            .call_kw(
                self.vararg.map_or(
                    pos_args.collect(list),
                    lambda v: pos_args.insert(v.forward_vararg(self.target)).collect(
                        list
                    ),
                ),
                keywords,
            )
        )
        if self.returns_relation:
            return Node.ref(Builtins.SELF).attr(NEW).call(call).node
        return call.into(_wrap, wrapper).node


class AstSignature(NamedTuple):
    name: str
    args: ast.arguments
    returns: ast.expr

    def _render(self, body: list[ast.stmt], decorators: list[ast.expr]) -> str:
        rendered = ast.FunctionDef(
            name=self.name,
            args=self.args,
            body=body,
            decorator_list=decorators,
            returns=self.returns,
        )
        ast.fix_missing_locations(rendered)
        return indent(ast.unparse(rendered), INDENT)

    def render(
        self,
        doc: str,
        return_expr: ast.expr,
        decorators: pc.Option[pc.Iter[str]] = pc.NONE,
    ) -> str:
        return self._render(
            pc.Option.if_some(doc.strip()).map_or(
                [ast.Return(return_expr)],
                lambda txt: [ast.Expr(ast.Constant(txt)), ast.Return(return_expr)],
            ),
            decorators.map(lambda ds: ds.iter().map(ast.Name).collect(list)).unwrap_or(
                []
            ),  # pyright: ignore[reportArgumentType]
        )

    def render_overload(self) -> str:
        return self._render(
            [ast.Expr(ast.Constant(Ellipsis))], [ast.Name(Typing.OVERLOAD)]
        )


def _to_expr(expr: str) -> ast.expr:
    return ast.parse(expr, mode="eval").body


def _make_arg(name: str, annotation: str) -> ast.arg:
    return ast.arg(name, _to_expr(annotation))
