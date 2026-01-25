"""Compare pql API coverage against Polars."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Self

import polars as pl
import pyochain as pc

import pql


class MatchStatus(Enum):
    """Status of a method comparison."""

    MISSING = auto()
    SIGNATURE_MISMATCH = auto()
    MATCH = auto()
    EXTRA = auto()


@dataclass(slots=True)
class ParamInfo:
    """Information about a function parameter."""

    name: str
    kind: str
    has_default: bool
    annotation: pc.Option[str]

    @classmethod
    def from_signature(cls, param: inspect.Parameter) -> Self:
        """Create ParamInfo from inspect.Parameter."""
        return cls(
            name=param.name,
            kind=str(param.kind),
            has_default=param.default is not inspect.Parameter.empty,
            annotation=_get_annotation_str(param.annotation),
        )


@dataclass(slots=True)
class MethodInfo:
    """Information about a method."""

    name: str
    params: pc.Seq[ParamInfo]
    return_annotation: pc.Option[str]
    is_property: bool = False

    def signature_str(self) -> str:
        """Generate a human-readable signature string."""
        params_str = (
            self.params.iter()
            .filter(lambda p: p.name != "self")
            .map(
                lambda p: p.annotation.map(lambda a: f"{p.name}: {a}").unwrap_or(
                    p.name + ("=..." if p.has_default else "")
                )
            )
            .join(", ")
        )
        ret = self.return_annotation.map(lambda r: f" -> {r}").unwrap_or("")
        return f"({params_str}){ret}"


@dataclass(slots=True)
class ComparisonResult:
    """Result of comparing a single method."""

    method_name: str
    status: MatchStatus
    polars_info: pc.Option[MethodInfo]
    pql_info: pc.Option[MethodInfo]
    diff_details: pc.Option[str] = field(default_factory=lambda: pc.NONE)

    @classmethod
    def from_method(cls, polars_cls: type, pql_cls: type, method_name: str) -> Self:
        """Compare a single method between polars and pql."""
        match (
            _get_method_info(polars_cls, method_name),
            _get_method_info(pql_cls, method_name),
        ):
            case (pc.NoneOption(), pc.Some(pql_info)):
                return cls(
                    method_name=method_name,
                    status=MatchStatus.EXTRA,
                    polars_info=pc.NONE,
                    pql_info=pc.Some(pql_info),
                )
            case (pc.Some(pl_info), pc.NoneOption()):
                return cls(
                    method_name=method_name,
                    status=MatchStatus.MISSING,
                    polars_info=pc.Some(pl_info),
                    pql_info=pc.NONE,
                )
            case (pc.Some(pl_info), pc.Some(pql_info)):
                diffs = _compare_params(pl_info.params, pql_info.params)
                status = (
                    MatchStatus.SIGNATURE_MISMATCH
                    if diffs.length() > 0
                    else MatchStatus.MATCH
                )
                return cls(
                    method_name=method_name,
                    status=status,
                    polars_info=pc.Some(pl_info),
                    pql_info=pc.Some(pql_info),
                    diff_details=diffs.then_some().map(lambda d: d.iter().join("; ")),
                )
            case _:
                return cls(
                    method_name=method_name,
                    status=MatchStatus.MISSING,
                    polars_info=pc.NONE,
                    pql_info=pc.NONE,
                )


def _getattr(obj: object, name: str) -> pc.Option[object]:
    """Safe getattr returning Option."""
    return pc.Option(getattr(obj, name, None))


def _compare_params(
    polars_params: pc.Seq[ParamInfo], pql_params: pc.Seq[ParamInfo]
) -> pc.Seq[str]:
    def _filter_self(params: pc.Seq[ParamInfo]) -> pc.Set[str]:
        return (
            params.iter()
            .filter(lambda p: p.name != "self")
            .map(lambda p: p.name)
            .collect(pc.Set)
        )

    pl_names = polars_params.into(_filter_self)
    pql_names = pql_params.into(_filter_self)

    return (
        pc.Iter(
            (
                pl_names.difference(pql_names)
                .then_some()
                .map(lambda m: f"Missing params: {m.iter().sort().join(', ')}"),
                pql_names.difference(pl_names)
                .then_some()
                .map(lambda e: f"Extra params: {e.iter().sort().join(', ')}"),
            )
        )
        .filter_map(lambda x: x)
        .collect()
    )


def _get_method_info(cls: type, name: str) -> pc.Option[MethodInfo]:
    def _build_method_info(attr: object) -> pc.Option[MethodInfo]:
        if isinstance(attr, property):
            return pc.Option(
                MethodInfo(
                    name=name,
                    params=pc.Seq(()),
                    return_annotation=pc.NONE,
                    is_property=True,
                )
            )
        if not callable(attr):
            return pc.NONE
        try:
            sig = inspect.signature(attr)
            return pc.Option(
                MethodInfo(
                    name=name,
                    params=pc.Iter(sig.parameters.values())
                    .map(ParamInfo.from_signature)
                    .collect(),
                    return_annotation=_get_annotation_str(sig.return_annotation),
                )
            )
        except (ValueError, TypeError):
            return pc.NONE

    return _getattr(cls, name).and_then(_build_method_info)


def _get_annotation_str(annotation: object) -> pc.Option[str]:
    """Convert annotation to string representation."""
    if annotation is inspect.Parameter.empty or annotation is inspect.Signature.empty:
        return pc.NONE
    match annotation:
        case type():
            return pc.Option(annotation.__name__)
        case _:
            return pc.Option(str(annotation))


@dataclass(slots=True)
class ClassComparison:
    """Comparison results for a class."""

    class_name: str
    results: pc.Seq[ComparisonResult]

    @classmethod
    def from_classes(
        cls, polars_cls: type, pql_cls: type, name: str | None = None
    ) -> Self:
        """Compare two classes and return comparison results."""

        def _get_public_methods(cls: type) -> pc.Set[str]:
            return (
                pc.Iter(dir(cls))
                .filter(lambda name: not name.startswith("_"))
                .filter(
                    lambda name: (
                        _getattr(cls, name)
                        .map(lambda attr: callable(attr) or isinstance(attr, property))
                        .unwrap_or(default=False)
                    )
                )
                .collect(pc.Set)
            )

        return cls(
            class_name=polars_cls.__name__ if name is None else name,
            results=(
                _get_public_methods(polars_cls)
                .union(_get_public_methods(pql_cls))
                .iter()
                .map(
                    lambda name: ComparisonResult.from_method(polars_cls, pql_cls, name)
                )
                .collect()
            ),
        )

    def by_status(self, status: MatchStatus) -> pc.Seq[ComparisonResult]:
        """Filter results by match status."""
        return self.results.iter().filter(lambda r: r.status == status).collect()

    def coverage_percent(self) -> float:
        """Calculate API coverage percentage."""
        total_polars = (
            self.results.iter().filter(lambda r: r.polars_info.is_some()).length()
        )
        if total_polars == 0:
            return 100.0
        matched = self.by_status(MatchStatus.MATCH).length()
        return (matched / total_polars) * 100


def generate_report(comparisons: pc.Seq[ClassComparison]) -> str:
    """Generate a markdown report from comparison results."""

    def _format_class_detail(comp: ClassComparison) -> pc.Seq[str]:
        """Format detailed sections for a class comparison."""
        return pc.Seq(
            (
                f"\n## {comp.class_name}\n",
                _format_section(
                    "[v] Matched Methods", comp.by_status(MatchStatus.MATCH)
                ),
                _format_section(
                    "[x] Missing Methods",
                    comp.by_status(MatchStatus.MISSING),
                    show_signature=True,
                ),
                _format_section(
                    "[!] Signature Mismatches",
                    comp.by_status(MatchStatus.SIGNATURE_MISMATCH),
                ),
                _format_section(
                    "[+] Extra Methods (pql-only)", comp.by_status(MatchStatus.EXTRA)
                ),
            )
        )

    def _format_section(
        title: str, items: pc.Seq[ComparisonResult], *, show_signature: bool = False
    ) -> str:
        """Format a section of the report."""
        count = items.length()
        if count == 0:
            return ""

        return (
            pc.Iter((f"\n### {title} ({count})\n",))
            .chain(
                items.iter().flat_map(
                    lambda r: _format_result_line(r, show_signature=show_signature)
                )
            )
            .join("\n")
        )

    def _format_result_line(
        result: ComparisonResult, *, show_signature: bool
    ) -> pc.Iter[str]:
        """Format a single comparison result as markdown lines."""
        if show_signature and result.polars_info.is_some():
            return pc.Iter.once(
                f"- `{result.method_name}` {result.polars_info.unwrap().signature_str()}"
            )
        if result.diff_details.is_some():
            return (
                pc.Iter.once(
                    f"- `{result.method_name}`: {result.diff_details.unwrap()}"
                )
                .iter()
                .chain(
                    result.polars_info.map(
                        lambda p: f"  - Polars: {p.signature_str()}"
                    ).iter()
                )
                .chain(
                    result.pql_info.map(
                        lambda p: f"  - pql: {p.signature_str()}"
                    ).iter()
                )
            )
        return pc.Iter.once(f"- `{result.method_name}`")

    def _format_summary_row(comp: ClassComparison) -> str:
        """Format a single summary table row."""
        return (
            f"| {comp.class_name} | "
            f"{comp.coverage_percent():.1f}% | "
            f"{comp.by_status(MatchStatus.MATCH).length()} | "
            f"{comp.by_status(MatchStatus.MISSING).length()} | "
            f"{comp.by_status(MatchStatus.SIGNATURE_MISMATCH).length()} | "
            f"{comp.by_status(MatchStatus.EXTRA).length()} |"
        )

    header: pc.Iter[str] = pc.Iter(
        (
            "# pql vs Polars API Comparison Report\n",
            "This report shows the API coverage of pql compared to Polars.\n",
            "## Summary\n",
            "| Class | Coverage | Matched | Missing | Mismatched | Extra |",
            "|-------|----------|---------|---------|------------|-------|",
        )
    )

    return (
        header.chain(comparisons.iter().map(_format_summary_row))
        .chain(comparisons.iter().flat_map(_format_class_detail))
        .join("\n")
    )


def main() -> None:
    """Run the comparison and generate report."""
    comparisons = pc.Seq(
        (
            ClassComparison.from_classes(pl.LazyFrame, pql.LazyFrame),
            ClassComparison.from_classes(pl.Expr, pql.Expr),
            ClassComparison.from_classes(
                pl.col("x").str.__class__, pql.Expr.str.__class__, name="Expr.str"
            ),
            ClassComparison.from_classes(
                pl.col("x").dt.__class__, pql.Expr.dt.__class__, name="Expr.dt"
            ),
        )
    ).into(generate_report)

    Path("API_COVERAGE.md").write_text(comparisons, encoding="utf-8")


if __name__ == "__main__":
    main()
