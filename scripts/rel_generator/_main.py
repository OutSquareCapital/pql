"""Generate a DuckDBPyRelation wrapper that uses SqlExpr instead of duckdb.Expression."""

from __future__ import annotations

from pathlib import Path

import pyochain as pc

from ._parse import extract_methods_from_stub
from ._raw import class_def, header
from ._structs import MethodInfo


def _resolve_overloads(methods: pc.Seq[MethodInfo]) -> pc.Seq[MethodInfo]:
    """For methods where ALL definitions are @overload, keep the last (most general) one as non-overload."""
    # Group by name, find groups that are *only* overloads
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

    def _resolve_one(m: MethodInfo) -> pc.Option[MethodInfo]:
        match (m.is_overload, overload_only_names.contains(m.name)):
            case (True, True):
                return pc.NONE
            case _:
                return pc.Some(m)

    return (
        methods.iter()
        .filter_map(_resolve_one)
        .chain(
            overload_only_names.iter().filter_map(
                lambda name: pc.Option(
                    methods.iter().filter(lambda m: m.name == name).last()
                ).map(
                    lambda m: MethodInfo(
                        name=m.name,
                        params=m.params,
                        vararg=m.vararg,
                        return_type=m.return_type,
                        is_overload=False,
                        is_property=m.is_property,
                        doc=m.doc,
                    )
                )
            )
        )
        .collect()
    )


def generate(stub_path: Path) -> str:
    """Generate the full ``_rel.py`` file content."""
    return (
        extract_methods_from_stub(stub_path)
        .into(_resolve_overloads)
        .iter()
        .filter_map(lambda m: m.generate_method())
        .into(lambda methods: f"{header()}{class_def()}{methods.join(chr(10) * 2)}\n")
    )
