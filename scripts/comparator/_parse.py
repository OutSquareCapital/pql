import ast
import re
from typing import override

import pyochain as pc

from .._utils import Builtins, Pql, Typing

GENERIC_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")

SELF_PATTERN = re.compile(r"\b(Self|Expr|LazyFrame)\b")


def _slice_to_expr(slice_node: ast.expr) -> ast.expr:
    match slice_node:
        case ast.Tuple() | ast.Name() | ast.Subscript() | ast.BinOp() | ast.Attribute():
            return slice_node
        case _:
            return ast.Name(id="Any", ctx=ast.Load())


class _AliasExpander(ast.NodeTransformer):
    _single_param_aliases: pc.Dict[str, str] = pc.Dict.from_kwargs(
        TryIter="Iterable[{arg}] | {arg}",
        TrySeq="Sequence[{arg}] | {arg}",
    )

    @override
    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        rewritten = self.generic_visit(node)
        match rewritten:
            case ast.Subscript(value=ast.Name(id=alias_name), slice=slice_node):
                return (
                    self._single_param_aliases.get_item(alias_name)
                    .map(
                        lambda template: ast.copy_location(
                            ast.parse(
                                template.format(
                                    arg=ast.unparse(_slice_to_expr(slice_node))
                                ),
                                mode="eval",
                            ).body,
                            rewritten,
                        )
                    )
                    .unwrap_or(rewritten)
                )
            case _:
                return rewritten


class _GenericCanonicalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._mapping: pc.Dict[str, str] = pc.Dict.new()

    @override
    def visit_Name(self, node: ast.Name) -> ast.AST:
        match node.id:
            case name if name not in {Typing.SELF, Pql.EXPR, Pql.LAZY_FRAME} and bool(
                GENERIC_SYMBOL_PATTERN.match(name)
            ):
                canonical_name = self._mapping.setdefault(
                    name, f"__GENERIC_{self._mapping.length()}__"
                )
                return ast.copy_location(
                    ast.Name(id=canonical_name, ctx=node.ctx), node
                )
            case _:
                return node


class _UnionCanonicalizer(ast.NodeTransformer):
    @override
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        match self.generic_visit(node):
            case ast.BinOp(op=ast.BitOr()):

                def _build_union_expr(parts: pc.Seq[str]) -> ast.expr:
                    def _union_expr(left: ast.expr, right: ast.expr) -> ast.expr:
                        return ast.BinOp(left=left, op=ast.BitOr(), right=right)

                    def _build_union(
                        acc: pc.Option[ast.expr], expr: ast.expr
                    ) -> pc.Option[ast.expr]:
                        return acc.map(lambda left: _union_expr(left, expr)).or_(
                            pc.Some(expr)
                        )

                    return (
                        parts.iter()
                        .map(lambda part: ast.parse(part, mode="eval").body)
                        .fold(pc.NONE, _build_union)
                        .unwrap_or(ast.Constant(value=None))
                    )

                def _union_members(node: ast.expr) -> pc.Iter[ast.expr]:
                    match node:
                        case ast.BinOp(left=left, op=ast.BitOr(), right=right):
                            return _union_members(left).chain(_union_members(right))
                        case _:
                            return pc.Iter.once(node)

                members_as_text = _union_members(node).map(ast.unparse).collect()
                has_float = members_as_text.iter().any(
                    lambda text: text == Builtins.FLOAT
                )

                return (
                    members_as_text.iter()
                    .filter(lambda text: not (has_float and text == Builtins.INT))
                    .unique()
                    .sort()
                    .into(_build_union_expr)
                )

            case _ as rewritten:
                return rewritten


def _parse_annotation(annotation: str) -> pc.Option[ast.Expression]:
    try:
        return pc.Some(ast.parse(annotation, mode="eval"))
    except SyntaxError:
        return pc.NONE


def _normalize(parsed: ast.Expression) -> str:
    return ast.unparse(
        ast.fix_missing_locations(  # pyright: ignore[reportAny]
            _UnionCanonicalizer().visit(  # pyright: ignore[reportAny]
                _GenericCanonicalizer().visit(_AliasExpander().visit(parsed))  # pyright: ignore[reportAny]
            )
        )
    )


def normalize_annotation(annotation: str) -> str:

    return extract_last_name(
        SELF_PATTERN.sub(
            "__SELF__",
            extract_last_name(
                _parse_annotation(annotation).map(_normalize).unwrap_or(annotation)
            ),
        )
    )


def extract_last_name(annotation: str) -> str:
    if "[" in annotation:
        base_type = annotation.split("[", maxsplit=1)[0]
        generic_part = annotation[len(base_type) :]
        return extract_last_name(base_type) + generic_part

    return annotation.rsplit(".", maxsplit=1)[-1]
