from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pyochain as pc

from ._rules import PYTYPING_REWRITES, SKIP_METHODS, TYPE_SUBS


class PyLit(StrEnum):
    SQLEXPR = "SqlExpr"
    DUCK_REL = "DuckDBPyRelation"
    DUCK_EXPR = "Expression"
    NONE = "None"
    ANY = "Any"
    SELF = "Self"


@dataclass(slots=True)
class ParamInfo:
    """A single parameter extracted from the stub."""

    name: str
    annotation: str
    default: pc.Option[str] = field(default_factory=lambda: pc.NONE)
    is_kw_only: bool = False

    def _rewrite_annotation(self) -> str:
        return _rewrite_type(self.annotation)

    def format_param(self) -> str:
        ann = self._rewrite_annotation()
        base = f"{self.name}: {ann}" if ann else self.name
        return self.default.map(lambda d: f"{base} = {d}").unwrap_or(base)

    def forward_vararg(self) -> str:
        """Generate the forwarded vararg, converting types at boundary."""
        match _rewrite_type(self.annotation):
            case a if PyLit.SQLEXPR.value in a and "str" not in a:
                return f"*(arg.inner() for arg in {self.name})"
            case a if PyLit.SQLEXPR.value in a:
                return (
                    f"*(arg.inner() if isinstance(arg, SqlExpr) else arg"
                    f" for arg in {self.name})"
                )
            case _:
                return f"*{self.name}"

    def forward_arg(self) -> str:
        """Generate the forwarded argument, converting types at boundary."""
        match self.annotation:
            case a if PyLit.DUCK_EXPR.value in a and PyLit.DUCK_REL.value not in a:
                return f"_expr_or({self.name})"
            case a if PyLit.DUCK_REL.value in a:
                return f"{self.name}.inner()"
            case _:
                return self.name


@dataclass(slots=True)
class MethodInfo:
    """A method extracted from the stub + runtime doc."""

    name: str
    params: pc.Vec[ParamInfo]
    vararg: pc.Option[ParamInfo]
    return_type: str
    is_overload: bool
    is_property: bool
    doc: str

    @property
    def returns_relation(self) -> bool:
        return self.return_type == PyLit.DUCK_REL.value

    @property
    def returns_none(self) -> bool:
        return self.return_type == PyLit.NONE.value

    def rewritten_return(self) -> str:
        return (
            PyLit.SELF.value
            if self.returns_relation
            else _rewrite_type(self.return_type)
        )

    def build_property(self) -> str:
        """Build a ``@property`` definition."""
        ret = self.rewritten_return()
        doc_str = _format_doc(self.doc).map(lambda d: f"\n{d}").unwrap_or("")
        return (
            f"    @property\n"
            f"    def {self.name}(self) -> {ret}:{doc_str}\n"
            f"        return self.inner().{self.name}"
        )

    def build_signature(self) -> str:
        """Build the full ``def`` signature line."""
        parts: pc.Vec[str] = pc.Vec(["self"])

        has_vararg = self.vararg.is_some()
        has_kw = self.params.iter().any(lambda p: p.is_kw_only)

        self.params.iter().filter(lambda p: not p.is_kw_only).for_each(
            lambda p: parts.append(p.format_param())
        )

        match (has_vararg, has_kw):
            case (True, _):
                self.vararg.inspect(
                    lambda v: parts.append(f"*{v.name}: {_rewrite_type(v.annotation)}")
                )
            case (False, True):
                parts.append("*")
            case _:
                pass

        ret = self.rewritten_return()
        joined = (
            self.params.iter()
            .filter(lambda p: p.is_kw_only)
            .map(lambda p: p.format_param())
            .collect_into(parts)
            .join(", ")
        )
        return f"    def {self.name}({joined}) -> {ret}:"

    def generate_method(self) -> pc.Option[str]:
        """Generate a single method wrapper."""
        match self.name:
            case _ if (
                SKIP_METHODS.contains(self.name)
                or self.name.startswith("__")
                or self.is_overload
            ):
                return pc.NONE
            case _ if self.is_property:
                return pc.Some(self.build_property())
            case _:
                sig = self.build_signature()
                doc_str = _format_doc(self.doc).map(lambda d: f"\n{d}").unwrap_or("")

                return pc.Some(f"{sig}{doc_str}\n{self.build_body()}")

    def build_body(self) -> str:
        """Build the method body that delegates to ``self._expr``."""
        regular = self.params.iter().filter(lambda p: not p.is_kw_only).collect(pc.Vec)
        kw_only = self.params.iter().filter(lambda p: p.is_kw_only).collect(pc.Vec)

        call_parts = pc.Vec[str].new()

        regular.iter().map(lambda p: p.forward_arg()).collect_into(call_parts)

        self.vararg.inspect(
            lambda v: call_parts.append(
                v.forward_vararg() if "Expression" in v.annotation else f"*{v.name}"
            )
        )

        kw_only.iter().for_each(
            lambda p: call_parts.append(f"{p.name}={p.forward_arg()}")
        )

        call = f"self.inner().{self.name}({call_parts.join(', ')})"

        if self.returns_relation:
            return f"        return self._new({call})"
        return f"        return {call}"


def _rewrite_type(annotation: str) -> str:
    """Rewrite a stub type annotation for use in the generated wrapper."""

    def _sub_one(acc: str, old_new: tuple[str, str]) -> str:
        return acc.replace(old_new[0], old_new[1])

    return (
        TYPE_SUBS.items()
        .iter()
        .fold(
            PYTYPING_REWRITES.items().iter().fold(annotation, _sub_one),
            _sub_one,
        )
    )


def _format_doc(doc: str) -> pc.Option[str]:
    """Format a docstring for the generated wrapper."""
    return (
        pc.Vec.from_ref(doc.strip().splitlines())
        .then_some()
        .map(
            lambda lines: (
                f'        """{lines.first().rstrip()}"""'
                if lines.length() == 1
                else (
                    f'        """{lines.first().rstrip()}\n'
                    + lines.iter()
                    .skip(1)
                    .map(lambda line: f"        {line}")
                    .join("\n")
                    + '\n        """'
                )
            )
        )
    )
