"""Compare pql API coverage against Polars."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from functools import partial
from pathlib import Path
from typing import NamedTuple, Self, cast

import narwhals as nw
import polars as pl
import pyochain as pc

import pql

type MapInfo = pc.Dict[str, ParamInfo]

SELF_PATTERN = re.compile(r"\b(Self|Expr|LazyFrame)\b")


def main() -> None:
    """Run the comparison and generate report."""
    comparisons = (
        pc.Iter(
            (
                ClassComparison(nw.LazyFrame, pl.LazyFrame, pql.LazyFrame),
                ClassComparison(nw.Expr, pl.Expr, pql.Expr),
                ClassComparison(
                    nw.col("x").str.__class__,
                    pl.col("x").str.__class__,
                    pql.col("x").str.__class__,
                    name="Expr.str",
                ),
                ClassComparison(
                    nw.col("x").list.__class__,
                    pl.col("x").list.__class__,
                    pql.col("x").list.__class__,
                    name="Expr.list",
                ),
                ClassComparison(
                    nw.col("x").struct.__class__,
                    pl.col("x").struct.__class__,
                    pql.col("x").struct.__class__,
                    name="Expr.struct",
                ),
            )
        )
        .map(lambda comp: comp.to_report())
        .collect()
        .into(
            lambda comps: (
                _header()
                .chain(_render_summary_table(comps))
                .chain(comps.iter().flat_map(lambda comp: comp.to_section()))
            )
        )
        .join("\n")
    )

    Path("API_COVERAGE.md").write_text(comparisons, encoding="utf-8")


def _header() -> pc.Iter[str]:
    return pc.Iter(
        (
            "# pql vs Polars API Comparison Report\n",
            "This report shows the API coverage of pql compared to Polars.\n",
            "## Summary\n",
        )
    )


def _render_summary_table(comps: pc.Seq[ComparisonReport]) -> pc.Iter[str]:
    data_rows = _summary_rows(comps)
    widths = (
        pc.Iter.once(_summary_header()).chain(data_rows).collect().into(_summary_widths)
    )
    return (
        pc.Iter.once(_format_row(_summary_header(), widths))
        .chain(pc.Iter.once(_format_separator(widths)))
        .chain(data_rows.iter().map(lambda row: _format_row(row, widths)))
    )


def _summary_header() -> pc.Seq[str]:
    return pc.Seq(
        (
            "Class",
            "Coverage vs Narwhals",
            "Total",
            "Matched",
            "Missing",
            "Mismatched",
            "Extra",
            "Extra vs Narwhals",
        )
    )


def _summary_rows(comps: pc.Seq[ComparisonReport]) -> pc.Seq[pc.Seq[str]]:
    return comps.iter().map(lambda comp: comp.to_row()).collect()


def _summary_widths(rows: pc.Seq[pc.Seq[str]]) -> pc.Seq[int]:
    return (
        pc.Iter(range(_summary_header().length()))
        .map(
            lambda idx: (
                rows.iter()
                .map(lambda row: len(row[idx]))
                .fold(0, lambda acc, length: max(length, acc))
            )
        )
        .collect()
    )


def _format_row(row: pc.Seq[str], widths: pc.Seq[int]) -> str:
    cells = (
        pc.Iter(range(widths.length()))
        .map(lambda idx: row[idx].ljust(widths[idx]))
        .join(" | ")
    )
    return f"| {cells} |"


def _format_separator(widths: pc.Seq[int]) -> str:
    cells = widths.iter().map(lambda width: "-" * width).join(" | ")
    return f"| {cells} |"


class Status(Enum):
    """Status of a method comparison."""

    MISSING = auto()
    SIGNATURE_MISMATCH = auto()
    MATCH = auto()
    EXTRA = auto()


class MismatchOn(StrEnum):
    """Source of a signature mismatch."""

    NW = auto()
    PL = auto()
    NULL = ""


@dataclass(slots=True)
class ParamInfo:
    """Information about a function parameter."""

    name: str
    has_default: bool
    annotation: pc.Option[str]

    @classmethod
    def from_signature(cls, param: inspect.Parameter) -> Self:
        """Create ParamInfo from inspect.Parameter."""
        return cls(
            name=param.name,
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

    def to_map(self) -> MapInfo:
        """Convert parameters to a dictionary mapping names to ParamInfo."""
        return (
            self.params.iter()
            .filter(lambda p: p.name != "self")
            .map(lambda p: (p.name, p))
            .collect(pc.Dict)
        )


def _get_annotation_str(annotation: object) -> pc.Option[str]:
    """Convert annotation to string representation."""
    match annotation:
        case inspect.Parameter.empty | inspect.Signature.empty:
            return pc.NONE
        case type():
            return pc.Option(annotation.__name__)
        case _:
            return pc.Option(str(annotation))


class MethodStatus(NamedTuple):
    """Status of a method comparison."""

    status: Status
    mismatch_source: MismatchOn


@dataclass(slots=True)
class ComparisonInfos:
    """Holds MethodInfo for narwhals, polars, and pql."""

    narwhals: pc.Option[MethodInfo] = field(default_factory=lambda: pc.NONE)
    polars: pc.Option[MethodInfo] = field(default_factory=lambda: pc.NONE)
    pql_info: pc.Option[MethodInfo] = field(default_factory=lambda: pc.NONE)

    def to_status(self) -> MethodStatus:  # noqa: PLR0911
        """Classify the method comparison result."""
        match (self.pql_info, self.narwhals, self.polars):
            case (pc.NONE, pc.Some(_), _):
                return MethodStatus(Status.MISSING, MismatchOn.NULL)
            case (pc.Some(_), pc.NONE, pc.NONE):
                return MethodStatus(Status.EXTRA, MismatchOn.NULL)
            case (pc.Some(target), pc.NONE, pc.Some(pl_info)):
                match _mismatch_against(target.to_map(), pl_info.to_map()):
                    case True:
                        return MethodStatus(Status.SIGNATURE_MISMATCH, MismatchOn.PL)
                    case False:
                        return MethodStatus(Status.MATCH, MismatchOn.NULL)
            case (pc.Some(target), pc.Some(nw_info), pc.Some(pl_info)):
                target_vs = partial(_mismatch_against, target.to_map())
                match target_vs(nw_info.to_map()) and target_vs(pl_info.to_map()):
                    case True:
                        return MethodStatus(Status.SIGNATURE_MISMATCH, MismatchOn.NW)
                    case False:
                        return MethodStatus(Status.MATCH, MismatchOn.NULL)
            case (pc.Some(target), pc.Some(nw_info), pc.NONE):
                return (
                    MethodStatus(Status.SIGNATURE_MISMATCH, MismatchOn.NW)
                    if _mismatch_against(target.to_map(), nw_info.to_map())
                    else MethodStatus(Status.MATCH, MismatchOn.NULL)
                )
            case _:
                return MethodStatus(Status.MISSING, MismatchOn.NULL)


def _mismatch_against(target: MapInfo, other: MapInfo) -> bool:
    on_params = other.keys().symmetric_difference(target.keys()).length() > 0
    on_ann = (
        other.keys()
        .intersection(target.keys())
        .any(
            lambda name: _annotations_differ(
                other.get_item(name).unwrap(),
                target.get_item(name).unwrap(),
            )
        )
    )
    return on_params or on_ann


def _normalize_self(annotation: str) -> str:
    return SELF_PATTERN.sub("__SELF__", annotation)


def _annotations_differ(pl_param: ParamInfo, pql_param: ParamInfo) -> bool:
    match (pl_param.annotation, pql_param.annotation):
        case (pc.Some(pl_ann), pc.Some(pql_ann)):
            return _normalize_self(pl_ann) != _normalize_self(pql_ann)
        case _:
            return False


def _diff_param_names(base: MethodInfo, other: MethodInfo) -> pc.Set[str]:
    base_map = base.to_map()
    other_map = other.to_map()
    return (
        base_map.keys()
        .symmetric_difference(other_map.keys())
        .iter()
        .chain(
            base_map.keys()
            .intersection(other_map.keys())
            .iter()
            .filter(
                lambda name: _annotations_differ(
                    base_map.get_item(name).unwrap(),
                    other_map.get_item(name).unwrap(),
                )
            )
        )
        .collect(pc.Set)
    )


def _signature_with_diff(base: MethodInfo, other: MethodInfo) -> str:
    diff_names = _diff_param_names(base, other)

    def _format_param(p: ParamInfo) -> str:
        match diff_names.any(lambda name: name == p.name):
            case True:
                return f"**{p.annotation.map(lambda a: f'{p.name}: {a}').unwrap_or(p.name + ('=...' if p.has_default else ''))}**"
            case False:
                return p.annotation.map(lambda a: f"{p.name}: {a}").unwrap_or(
                    p.name + ("=..." if p.has_default else "")
                )

    params_str = (
        base.params.iter()
        .filter(lambda p: p.name != "self")
        .map(_format_param)
        .join(", ")
    )
    ret = base.return_annotation.map(lambda r: f" -> {r}").unwrap_or("")
    return f"({params_str}){ret}"


@dataclass(slots=True)
class ComparisonResult:
    """Result of comparing a single method."""

    method_name: str
    classification: MethodStatus
    infos: ComparisonInfos

    @classmethod
    def from_method(
        cls, narwhals_cls: type, polars_cls: type, pql_cls: type, method_name: str
    ) -> Self:
        """Compare a single method between narwhals, polars, and pql."""
        infos = ComparisonInfos(
            narwhals=_get_method_info(narwhals_cls, method_name),
            polars=_get_method_info(polars_cls, method_name),
            pql_info=_get_method_info(pql_cls, method_name),
        )
        return cls(
            method_name=method_name,
            classification=infos.to_status(),
            infos=infos,
        )

    def to_format(self, *, status: Status) -> pc.Iter[str]:
        """Format a single comparison result as markdown lines."""
        match (status, self.infos.narwhals, self.infos.polars, self.infos.pql_info):
            case (Status.MISSING, pc.Some(narwhals_info), _, _):
                return pc.Iter.once(
                    f"- `{self.method_name}` {narwhals_info.signature_str()}"
                )
            case (
                Status.SIGNATURE_MISMATCH,
                pc.Some(nw_info),
                _,
                pc.Some(pql_info),
            ):
                return (
                    pc.Iter(
                        (
                            f"- `{self.method_name}` ({self.classification.mismatch_source.value})",
                            f"  - {'Narwhals'}: {_signature_with_diff(nw_info, pql_info)}",
                        )
                    )
                    .chain(
                        self.infos.polars.map(
                            lambda pl_info: pc.Iter.once(
                                f"  - {'Polars'}: {_signature_with_diff(pl_info, pql_info)}"
                            )
                        ).unwrap_or(default=pc.Iter(()))
                    )
                    .chain(
                        pc.Iter.once(
                            f"  - pql: {_signature_with_diff(pql_info, nw_info)}"
                        )
                    )
                )
            case (
                Status.SIGNATURE_MISMATCH,
                pc.NONE,
                pc.Some(pl_info),
                pc.Some(pql_info),
            ):
                return pc.Iter(
                    (
                        f"- `{self.method_name}` ({self.classification.mismatch_source.value})",
                        f"  - {'Polars'}: {_signature_with_diff(pl_info, pql_info)}",
                        f"  - pql: {_signature_with_diff(pql_info, pl_info)}",
                    )
                )
            case _:
                return pc.Iter.once(f"- `{self.method_name}`")


def _getattr(obj: object, name: str) -> pc.Option[object]:
    """Safe getattr returning Option."""
    return pc.Option(getattr(obj, name, None))


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
class ComparisonReport:
    """Report for a class comparison."""

    name: str
    results: pc.Vec[ComparisonResult]

    def to_section(self) -> pc.Seq[str]:
        """Format detailed sections for a class comparison."""
        return pc.Seq(
            (
                f"\n## {self.name}\n",
                _format(self.results, "[v] Matched Methods", status=Status.MATCH),
                _format(self.results, "[x] Missing Methods", status=Status.MISSING),
                _format(
                    self.results,
                    "[!] Signature Mismatches",
                    status=Status.SIGNATURE_MISMATCH,
                ),
                _format(
                    self.results, "[+] Extra Methods (pql-only)", status=Status.EXTRA
                ),
            )
        )

    def to_row(self) -> pc.Seq[str]:
        """Return a row of summary data as columns."""
        return pc.Seq(
            (
                self.name,
                f"{_coverage_percent(self.results):.1f}%",
                str(_total_methods(self.results)),
                str(_by_status(self.results, Status.MATCH).length()),
                str(_by_status(self.results, Status.MISSING).length()),
                str(_by_status(self.results, Status.SIGNATURE_MISMATCH).length()),
                str(_by_status(self.results, Status.EXTRA).length()),
                str(_extra_vs_narwhals(self.results).length()),
            )
        )


@dataclass(slots=True)
class ClassComparison:
    """Converter between entry arguments and ComparisonReport."""

    narwhals_cls: type
    polars_cls: type
    pql_cls: type
    name: str = ""

    def __post_init__(self) -> None:
        """Set default name if not provided."""
        if self.name == "":
            self.name = self.narwhals_cls.__name__

    def to_report(self) -> ComparisonReport:
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

        return ComparisonReport(
            self.name,
            _get_public_methods(self.narwhals_cls)
            .union(_get_public_methods(self.pql_cls))
            .iter()
            .map(
                lambda name: ComparisonResult.from_method(
                    self.narwhals_cls, self.polars_cls, self.pql_cls, name
                )
            )
            .sort(key=lambda r: r.method_name),
        )


def _format(results: pc.Vec[ComparisonResult], title: str, *, status: Status) -> str:
    """Format a section of the report."""
    items = _by_status(results, status)
    match items.length():
        case 0:
            return ""
        case count:
            return (
                pc.Iter((f"\n### {title} ({count})\n",))
                .chain(items.iter().flat_map(lambda r: r.to_format(status=status)))
                .join("\n")
            )


def _coverage_percent(results: pc.Vec[ComparisonResult]) -> float:
    """Calculate API coverage percentage."""

    def _accumulator(acc: tuple[int, int], r: ComparisonResult) -> tuple[int, int]:
        return (
            acc[0]
            + (1 if r.infos.narwhals.is_some() or r.infos.polars.is_some() else 0),
            acc[1] + (1 if r.classification.status == Status.MATCH else 0),
        )

    match results.iter().fold(cast(tuple[int, int], (0, 0)), _accumulator):
        case (0, _):
            return 100.0
        case (total_narwhals, matched):
            return (matched / total_narwhals) * 100


def _total_methods(results: pc.Vec[ComparisonResult]) -> int:
    """Count total methods considered for coverage."""
    return (
        results.iter()
        .filter(lambda r: r.infos.narwhals.is_some() or r.infos.polars.is_some())
        .length()
    )


def _by_status(
    results: pc.Vec[ComparisonResult], status: Status
) -> pc.Seq[ComparisonResult]:
    """Filter results by match status."""
    return results.iter().filter(lambda r: r.classification.status == status).collect()


def _extra_vs_narwhals(results: pc.Vec[ComparisonResult]) -> pc.Seq[ComparisonResult]:
    """Filter results where pql has methods missing in narwhals."""
    return (
        results.iter()
        .filter(
            lambda r: (
                r.infos.pql_info.is_some()
                and r.infos.narwhals.is_none()
                and r.infos.polars.is_some()
            )
        )
        .collect()
    )


if __name__ == "__main__":
    main()
