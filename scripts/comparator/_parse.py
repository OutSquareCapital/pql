import ast
import re

import pyochain as pc

GENERIC_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")

SELF_PATTERN = re.compile(r"\b(Self|Expr|LazyFrame)\b")


class _GenericCanonicalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._mapping: dict[str, str] = {}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        match node.id:
            case name if name not in {"Self", "Expr", "LazyFrame"} and bool(
                GENERIC_SYMBOL_PATTERN.match(name)
            ):
                canonical_name = self._mapping.setdefault(
                    name, f"__GENERIC_{len(self._mapping)}__"
                )
                return ast.copy_location(
                    ast.Name(id=canonical_name, ctx=node.ctx), node
                )
            case _:
                return node


class _UnionCanonicalizer(ast.NodeTransformer):
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
                has_float = members_as_text.iter().any(lambda text: text == "float")

                return (
                    members_as_text.iter()
                    .filter(lambda text: not (has_float and text == "int"))
                    .collect(pc.Set)
                    .iter()
                    .sort()
                    .into(_build_union_expr)
                )

            case _ as rewritten:
                return rewritten


def normalize_annotation(annotation: str) -> str:
    def _normalize_unions(annotation: str) -> str:
        try:
            parsed = ast.parse(annotation, mode="eval")
        except SyntaxError:
            return annotation

        normalized = _UnionCanonicalizer().visit(parsed)
        ast.fix_missing_locations(normalized)
        return ast.unparse(normalized)

    return extract_last_name(
        SELF_PATTERN.sub(
            "__SELF__",
            extract_last_name(_normalize_unions(_normalize_generics(annotation))),
        )
    )


def _normalize_generics(annotation: str) -> str:
    try:
        parsed = ast.parse(annotation, mode="eval")
    except SyntaxError:
        return annotation

    normalized = _GenericCanonicalizer().visit(parsed)
    ast.fix_missing_locations(normalized)
    return ast.unparse(normalized)


def extract_last_name(annotation: str) -> str:
    if "[" in annotation:
        base_type = annotation.split("[", maxsplit=1)[0]
        generic_part = annotation[len(base_type) :]
        return extract_last_name(base_type) + generic_part

    return annotation.rsplit(".", maxsplit=1)[-1]
