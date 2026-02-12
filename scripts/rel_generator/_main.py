"""Generate a DuckDBPyRelation wrapper that uses SqlExpr instead of duckdb.Expression."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyochain as pc

from ._parse import extract_methods_from_stub
from ._raw import class_def, header

if TYPE_CHECKING:
    from ._structs import MethodInfo


def _resolve_overloads(methods: pc.Seq[MethodInfo]) -> pc.Iter[MethodInfo]:
    """For methods where ALL definitions are @overload, add a non-overload implementation."""
    overload_only_names: pc.Set[str] = (
        methods.iter()
        .filter(lambda m: m.is_overload)
        .map(lambda m: m.name)
        .collect(pc.Set)
        .difference(
            methods.iter()
            .filter(lambda m: not m.is_overload and not m.is_property)
            .map(lambda m: m.name)
            .collect(pc.Set)
        )
    )

    def _is_last_overload(idx: int, method: MethodInfo) -> bool:
        return (
            not methods.iter()
            .skip(idx + 1)
            .any(lambda other: other.name == method.name and other.is_overload)
        )

    def _expand(idx: int, method: MethodInfo) -> MethodInfo:
        if (
            method.is_overload
            and overload_only_names.contains(method.name)
            and _is_last_overload(idx, method)
        ):
            return method.as_impl()
        return method

    return methods.iter().enumerate().map_star(_expand)


def generate(stub_path: Path) -> str:
    """Generate the full ``_rel.py`` file content."""
    return (
        extract_methods_from_stub(stub_path)
        .into(_resolve_overloads)
        .map(lambda m: m.generate_method())
        .into(lambda methods: f"{header()}{class_def()}{methods.join(chr(10) * 2)}\n")
    )
