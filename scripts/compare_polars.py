"""Compare pql API coverage against Polars."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from pathlib import Path
from typing import Self, cast

import narwhals as nw
import polars as pl
import pyochain as pc

import pql

SELF_PATTERN = re.compile(r"\b(Self|Expr|LazyFrame)\b")


class MatchStatus(Enum):
    """Status of a method comparison."""

    MISSING = auto()
    SIGNATURE_MISMATCH = auto()
    MATCH = auto()
    EXTRA = auto()


class MismatchSource(StrEnum):
    """Source of a signature mismatch."""

    NW = auto()
    PL = auto()
    NULL = ""


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

    @classmethod
    def from_signature(cls, name: str, sig: inspect.Signature) -> Self:
        """Create MethodInfo from inspect.Signature."""
        return cls(
            name=name,
            params=pc.Iter(sig.parameters.values())
            .map(ParamInfo.from_signature)
            .collect(),
            return_annotation=_get_annotation_str(sig.return_annotation),
        )

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


def _get_annotation_str(annotation: object) -> pc.Option[str]:
    """Convert annotation to string representation."""
    match annotation:
        case inspect.Parameter.empty | inspect.Signature.empty:
            return pc.NONE
        case type():
            return pc.Option(annotation.__name__)
        case _:
            return pc.Option(str(annotation))


@dataclass(slots=True)
class ComparisonResult:
    """Result of comparing a single method."""

    method_name: str
    status: MatchStatus
    mismatch_source: MismatchSource
    narwhals_info: pc.Option[MethodInfo] = field(default_factory=lambda: pc.NONE)
    polars_info: pc.Option[MethodInfo] = field(default_factory=lambda: pc.NONE)
    pql_info: pc.Option[MethodInfo] = field(default_factory=lambda: pc.NONE)

    @classmethod
    def from_method(
        cls, narwhals_cls: type, polars_cls: type, pql_cls: type, method_name: str
    ) -> Self:
        """Compare a single method between narwhals, polars, and pql."""
        polars_info = _get_method_info(polars_cls, method_name)
        match (
            _get_method_info(narwhals_cls, method_name),
            _get_method_info(pql_cls, method_name),
        ):
            case (pc.NONE, pc.Some(pql_method)):
                return cls(
                    method_name=method_name,
                    status=MatchStatus.EXTRA,
                    polars_info=polars_info,
                    pql_info=pc.Some(pql_method),
                    mismatch_source=MismatchSource.NULL,
                )
            case (pc.Some(nw_info), pc.NONE):
                return cls(
                    method_name=method_name,
                    status=MatchStatus.MISSING,
                    narwhals_info=pc.Some(nw_info),
                    polars_info=polars_info,
                    mismatch_source=MismatchSource.NULL,
                )
            case (pc.Some(nw_info), pc.Some(pql_method)):
                match _has_param_mismatch(
                    nw_info.params, pql_method.params
                ) and polars_info.map(
                    lambda pl_info: _has_param_mismatch(
                        pl_info.params, pql_method.params
                    )
                ).unwrap_or(default=True):
                    case True:
                        return cls(
                            method_name=method_name,
                            status=MatchStatus.SIGNATURE_MISMATCH,
                            narwhals_info=pc.Some(nw_info),
                            polars_info=polars_info,
                            pql_info=pc.Some(pql_method),
                            mismatch_source=MismatchSource.NW,
                        )
                    case False:
                        return cls(
                            method_name=method_name,
                            status=MatchStatus.MATCH,
                            narwhals_info=pc.Some(nw_info),
                            polars_info=polars_info,
                            pql_info=pc.Some(pql_method),
                            mismatch_source=MismatchSource.NULL,
                        )
            case _:
                return cls(
                    method_name=method_name,
                    status=MatchStatus.MISSING,
                    mismatch_source=MismatchSource.NULL,
                )


def _getattr(obj: object, name: str) -> pc.Option[object]:
    """Safe getattr returning Option."""
    return pc.Option(getattr(obj, name, None))


def _has_param_mismatch(
    polars_params: pc.Seq[ParamInfo], pql_params: pc.Seq[ParamInfo]
) -> bool:
    def _normalize_self(annotation: str) -> str:
        return SELF_PATTERN.sub("__SELF__", annotation)

    def _param_map(params: pc.Seq[ParamInfo]) -> pc.Dict[str, ParamInfo]:
        return (
            params.iter()
            .filter(lambda p: p.name != "self")
            .map(lambda p: (p.name, p))
            .collect(pc.Dict)
        )

    def _annotations_differ(pl_param: ParamInfo, pql_param: ParamInfo) -> bool:
        match (pl_param.annotation, pql_param.annotation):
            case (pc.Some(pl_ann), pc.Some(pql_ann)):
                return _normalize_self(pl_ann) != _normalize_self(pql_ann)
            case _:
                return False

    def _check(
        pl_map: pc.Dict[str, ParamInfo], pql_map: pc.Dict[str, ParamInfo]
    ) -> bool:
        on_params = pl_map.keys().symmetric_difference(pql_map.keys()).length() > 0
        on_ann = (
            pl_map.keys()
            .intersection(pql_map.keys())
            .any(
                lambda name: _annotations_differ(
                    pl_map.get_item(name).unwrap(),
                    pql_map.get_item(name).unwrap(),
                )
            )
        )
        return on_params or on_ann

    return _check(polars_params.into(_param_map), pql_params.into(_param_map))


def _get_method_info(cls: type, name: str) -> pc.Option[MethodInfo]:
    return _getattr(cls, name).and_then(_build_method_info, name)


def _build_method_info(attr: object, name: str) -> pc.Option[MethodInfo]:
    match attr:
        case property():
            return pc.Some(
                MethodInfo(
                    name=name,
                    params=pc.Seq[ParamInfo].new(),
                    return_annotation=pc.NONE,
                    is_property=True,
                )
            )
        case attr if callable(attr):
            try:
                return pc.Some(
                    MethodInfo.from_signature(name=name, sig=inspect.signature(attr))
                )
            except (ValueError, TypeError):
                return pc.NONE
        case _:
            return pc.NONE


@dataclass(slots=True)
class ClassComparison:
    """Comparison results for a class."""

    class_name: str
    results: pc.Seq[ComparisonResult]

    @classmethod
    def from_classes(
        cls,
        narwhals_cls: type,
        polars_cls: type,
        pql_cls: type,
        name: str | None = None,
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
            class_name=narwhals_cls.__name__ if name is None else name,
            results=(
                _get_public_methods(narwhals_cls)
                .union(_get_public_methods(pql_cls))
                .iter()
                .map(
                    lambda name: ComparisonResult.from_method(
                        narwhals_cls, polars_cls, pql_cls, name
                    )
                )
                .sort(key=lambda r: r.method_name)
            ),
        )

    def by_status(self, status: MatchStatus) -> pc.Seq[ComparisonResult]:
        """Filter results by match status."""
        return self.results.iter().filter(lambda r: r.status == status).collect()

    def coverage_percent(self) -> float:
        """Calculate API coverage percentage."""
        match self.results.iter().fold(
            cast(tuple[int, int], (0, 0)),
            lambda acc, r: (
                acc[0] + r.narwhals_info.map(lambda _: 1).unwrap_or(default=0),
                acc[1] + (1 if r.status == MatchStatus.MATCH else 0),
            ),
        ):
            case (0, _):
                return 100.0
            case (total_narwhals, matched):
                return (matched / total_narwhals) * 100

    def to_section(self) -> pc.Seq[str]:
        """Format detailed sections for a class comparison."""
        return pc.Seq(
            (
                f"\n## {self.class_name}\n",
                self._format("[v] Matched Methods", status=MatchStatus.MATCH),
                self._format("[x] Missing Methods", status=MatchStatus.MISSING),
                self._format(
                    "[!] Signature Mismatches", status=MatchStatus.SIGNATURE_MISMATCH
                ),
                self._format("[+] Extra Methods (pql-only)", status=MatchStatus.EXTRA),
            )
        )

    def _format(self, title: str, *, status: MatchStatus) -> str:
        """Format a section of the report."""
        items = self.by_status(status)
        match items.length():
            case 0:
                return ""
            case count:
                return (
                    pc.Iter((f"\n### {title} ({count})\n",))
                    .chain(
                        items.iter().flat_map(
                            lambda r: _format_result_line(r, status=status)
                        )
                    )
                    .join("\n")
                )


def _format_result_line(
    result: ComparisonResult, *, status: MatchStatus
) -> pc.Iter[str]:
    """Format a single comparison result as markdown lines."""
    match (status, result.narwhals_info, result.pql_info):
        case (MatchStatus.MISSING, pc.Some(narwhals_info), _):
            return pc.Iter.once(
                f"- `{result.method_name}` {narwhals_info.signature_str()}"
            )
        case (MatchStatus.SIGNATURE_MISMATCH, pc.Some(nw_info), pc.Some(pql_info)):
            return (
                pc.Iter(
                    (
                        f"- `{result.method_name}` ({result.mismatch_source.value})",
                        f"  - {'Narwhals'}: {nw_info.signature_str()}",
                    )
                )
                .chain(
                    result.polars_info.map(
                        lambda pl_info: pc.Iter.once(
                            f"  - {'Polars'}: {pl_info.signature_str()}"
                        )
                    ).unwrap_or(default=pc.Iter(()))
                )
                .chain(pc.Iter.once(f"  - pql: {pql_info.signature_str()}"))
            )
        case _:
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


def _header() -> pc.Iter[str]:
    return pc.Iter(
        (
            "# pql vs Polars API Comparison Report\n",
            "This report shows the API coverage of pql compared to Polars.\n",
            "## Summary\n",
            "| Class | Coverage | Matched | Missing | Mismatched | Extra |",
            "|-------|----------|---------|---------|------------|-------|",
        )
    )


def main() -> None:
    """Run the comparison and generate report."""
    comparisons = (
        pc.Seq(
            (
                ClassComparison.from_classes(nw.LazyFrame, pl.LazyFrame, pql.LazyFrame),
                ClassComparison.from_classes(nw.Expr, pl.Expr, pql.Expr),
                ClassComparison.from_classes(
                    nw.col("x").str.__class__,
                    pl.col("x").str.__class__,
                    pql.col("x").str.__class__,
                    name="Expr.str",
                ),
            )
        )
        .into(
            lambda comps: (
                _header()
                .chain(comps.iter().map(_format_summary_row))
                .chain(comps.iter().flat_map(lambda comp: comp.to_section()))
            )
        )
        .join("\n")
    )

    Path("API_COVERAGE.md").write_text(comparisons, encoding="utf-8")


if __name__ == "__main__":
    main()
