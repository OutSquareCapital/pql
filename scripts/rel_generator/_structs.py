from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import pyochain as pc

from ._rules import PYTYPING_REWRITES, TYPE_SUBS, PyLit


@dataclass(slots=True)
class ParamInfo:
    """A single parameter extracted from the stub."""

    name: str
    annotation: str
    default: pc.Option[str] = field(default_factory=lambda: pc.NONE)
    is_kw_only: bool = False

    def format_param(self) -> str:
        ann = _rewrite_type(self.annotation)
        base = f"{self.name}: {ann}" if ann else self.name
        return self.default.map(lambda d: f"{base} = {d}").unwrap_or(base)

    def forward_vararg(self) -> str:
        """Generate the forwarded vararg, converting types at boundary."""
        match _rewrite_type(self.annotation):
            case a if PyLit.SQLEXPR in a and PyLit.STR not in a:
                return f"*(map(lambda arg: arg.inner(), {self.name}))"
            case a if PyLit.SQLEXPR in a:
                return f"*(map({PyLit.INTO_DUCKDB}, {self.name}))"
            case _:
                return f"*{self.name}"

    def forward_arg(self) -> str:
        """Generate the forwarded argument, converting types at boundary."""
        match self.annotation:
            case a if PyLit.ITERABLE in a and PyLit.DUCK_EXPR in a:
                return f"try_iter({self.name}).map({PyLit.INTO_DUCKDB})"
            case a if PyLit.DUCK_EXPR in a and PyLit.DUCK_REL not in a:
                return f"{PyLit.INTO_DUCKDB}({self.name})"
            case a if PyLit.DUCK_REL in a:
                return f"{self.name}.inner()"
            case _:
                return self.name


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

    def as_impl(self) -> Self:
        self.is_overload = False
        return self

    @property
    def returns_relation(self) -> bool:
        return self.return_type == PyLit.DUCK_REL

    @property
    def returns_none(self) -> bool:
        return self.return_type == PyLit.NONE

    def _return_meta(self) -> tuple[str, pc.Option[str]]:
        if self.returns_relation:
            return PyLit.SELF_RET, pc.NONE
        return _return_meta(self.return_type)

    def _build_signature(self, return_annotation: str) -> str:
        """Build the full ``def`` signature line."""
        parts: pc.Vec[str] = (
            self.params.iter()
            .filter(lambda p: not p.is_kw_only)
            .map(lambda p: p.format_param())
            .insert(PyLit.SELF)
            .collect(pc.Vec)
        )

        match (self.vararg.is_some(), self.params.iter().any(lambda p: p.is_kw_only)):
            case (True, _):
                self.vararg.inspect(
                    lambda v: parts.append(f"*{v.name}: {_rewrite_type(v.annotation)}")
                )
            case (False, True):
                parts.append("*")
            case _:
                pass

        joined = (
            self.params.iter()
            .filter(lambda p: p.is_kw_only)
            .map(lambda p: p.format_param())
            .collect_into(parts)
            .join(", ")
        )
        return f"    def {self.name}({joined}) -> {return_annotation}:"

    def generate_method(self) -> str:
        """Generate a single method wrapper."""
        return_annotation, wrapper = self._return_meta()
        match self.name:
            case _ if self.is_overload:
                return self._to_overload(return_annotation)
            case _ if self.is_property:
                return self._to_property(return_annotation, wrapper)
            case _:
                sig = self._build_signature(return_annotation)
                doc_str = _format_doc(self.doc)

                return f"{sig}{doc_str}\n{self._build_body(wrapper)}"

    def _to_overload(self, return_annotation: str) -> str:
        return f"    @{PyLit.OVERLOAD}\n{self._build_signature(return_annotation)}\n        ..."

    def _to_property(self, return_annotation: str, wrapper: pc.Option[str]) -> str:
        doc_str = _format_doc(self.doc)
        body = _wrap_return(f"self.inner().{self.name}", wrapper)
        return (
            f"    @{PyLit.PROPERTY}\n"
            f"    def {self.name}(self) -> {return_annotation}:{doc_str}\n"
            f"        return {body}"
        )

    def _build_body(self, wrapper: pc.Option[str]) -> str:
        """Build the method body that delegates to ``self._expr``."""
        call_parts = (
            self.params.iter()
            .filter(lambda p: not p.is_kw_only)
            .map(lambda p: p.forward_arg())
            .collect(pc.Vec)
        )

        self.vararg.map(
            lambda v: call_parts.append(
                v.forward_vararg() if PyLit.DUCK_EXPR in v.annotation else f"*{v.name}"
            )
        )

        all_args = (
            self.params.iter()
            .filter(lambda p: p.is_kw_only)
            .map(lambda p: f"{p.name}={p.forward_arg()}")
            .collect_into(call_parts)
            .join(", ")
        )

        call = f"self.inner().{self.name}({all_args})"

        if self.returns_relation:
            return f"        return self._new({call})"
        return f"        return {_wrap_return(call, wrapper)}"


def _rewrite_type(annotation: str) -> str:
    """Rewrite a stub type annotation for use in the generated wrapper."""

    def _sub_one(acc: str, old_new: tuple[str, str]) -> str:
        return acc.replace(old_new[0], old_new[1])

    return (
        TYPE_SUBS.items()
        .iter()
        .fold(PYTYPING_REWRITES.items().iter().fold(annotation, _sub_one), _sub_one)
    )


def _wrap_return(value: str, wrapper: pc.Option[str]) -> str:
    return wrapper.map(lambda fn_name: f"{fn_name}({value})").unwrap_or(value)


def _return_meta(annotation: str) -> tuple[str, pc.Option[str]]:

    def _collection_kind() -> pc.Option[PyLit]:
        match rewritten:
            case _ if rewritten.startswith(PyLit.LIST):
                return pc.Some(PyLit.LIST)
            case _ if rewritten.startswith(PyLit.DICT):
                return pc.Some(PyLit.DICT)
            case _:
                return pc.NONE

    rewritten = _rewrite_type(annotation)
    match _collection_kind():
        case pc.Some(PyLit.LIST):
            suffix = rewritten.removeprefix(PyLit.LIST)
            return (
                f"pc.Vec{suffix}" if suffix else "pc.Vec[Any]",
                pc.Some("pc.Vec.from_ref"),
            )
        case pc.Some(PyLit.DICT):
            suffix = rewritten.removeprefix(PyLit.DICT)
            return (
                f"pc.Dict{suffix}" if suffix else "pc.Dict[Any, Any]",
                pc.Some("pc.Dict.from_ref"),
            )
        case _:
            return rewritten, pc.NONE


def _format_doc(doc: str) -> str:
    """Format a docstring for the generated wrapper."""

    def _format_lines(lines: pc.Vec[str]) -> str:
        match lines.length():
            case 1:
                return f'        """{lines.first().rstrip()}"""'
            case _:
                first_ln = f'        """{lines.first().rstrip()}\n'
                last_ln = '\n        """'
                return (
                    first_ln
                    + lines.iter()
                    .skip(1)
                    .map(lambda line: f"        {line}")
                    .join("\n")
                    + last_ln
                )

    return (
        pc.Vec.from_ref(doc.strip().splitlines())
        .then_some()
        .map(_format_lines)
        .map(lambda d: f"\n{d}")
        .unwrap_or("")
    )
