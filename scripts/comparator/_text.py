from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pyochain as pc

from ._infos import (
    ComparisonResult,
    MethodInfo,
    ParamInfo,
    annotations_differ,
    get_attr,
)
from ._models import Status

type RefBackend = Literal["narwhals", "polars"]


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
        libs = LibCell()
        return pc.Seq(
            (
                self.name,
                libs.coverage(self.results),
                libs.count(self.results, _has_reference),
                libs.status(self.results, Status.MATCH),
                libs.status(self.results, Status.MISSING),
                libs.status(self.results, Status.SIGNATURE_MISMATCH),
                libs.status(self.results, Status.EXTRA),
            )
        )


def header() -> pc.Iter[str]:
    return pc.Iter(
        (
            "# pql vs Polars API Comparison Report\n",
            "This report shows the API coverage of pql compared to Polars.\n",
            "## Summary\n",
        )
    )


def _summary_header() -> pc.Seq[str]:
    return pc.Seq(
        (
            "Class",
            "Coverage",
            "Total",
            "Matched",
            "Missing",
            "Mismatched",
            "Extra",
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
                        get_attr(cls, name)
                        .map(lambda attr: callable(attr) or isinstance(attr, property))
                        .unwrap_or(default=False)
                    )
                )
                .collect(pc.Set)
            )

        return ComparisonReport(
            self.name,
            _get_public_methods(self.narwhals_cls)
            .union(_get_public_methods(self.polars_cls))
            .union(_get_public_methods(self.pql_cls))
            .iter()
            .map(
                lambda name: ComparisonResult.from_method(
                    self.narwhals_cls, self.polars_cls, self.pql_cls, name, self.name
                )
            )
            .sort(key=lambda r: r.method_name),
        )


def render_summary_table(comps: pc.Seq[ComparisonReport]) -> pc.Iter[str]:
    data_rows = _summary_rows(comps)

    return (
        pc.Iter.once(_summary_header())
        .chain(data_rows)
        .collect()
        .into(_summary_widths)
        .collect()
        .into(
            lambda widths: pc.Iter.once(_format_row(_summary_header(), widths)).chain(
                pc.Iter.once(_format_separator(widths)),
                data_rows.iter().map(lambda row: _format_row(row, widths)),
            )
        )
    )


def _summary_rows(comps: pc.Seq[ComparisonReport]) -> pc.Seq[pc.Seq[str]]:
    return comps.iter().map(lambda comp: comp.to_row()).collect()


def _summary_widths(rows: pc.Seq[pc.Seq[str]]) -> pc.Iter[int]:
    return pc.Iter(range(_summary_header().length())).map(
        lambda idx: (
            rows.iter()
            .map(lambda row: len(row[idx]))
            .fold(0, lambda acc, length: max(length, acc))
        )
    )


def _format_separator(widths: pc.Seq[int]) -> str:
    cells = widths.iter().map(lambda width: "-" * width).join(" | ")
    return f"| {cells} |"


def _format(results: pc.Vec[ComparisonResult], title: str, *, status: Status) -> str:
    """Format a section of the report."""
    return (
        results.into(_by_status, status)
        .then(
            lambda items: (
                pc.Iter((f"\n### {title} ({items.length()})\n",))
                .chain(items.iter().flat_map(lambda r: r.to_format(status=status)))
                .join("\n")
            )
        )
        .unwrap_or("")
    )


def _format_row(row: pc.Seq[str], widths: pc.Seq[int]) -> str:
    cells = (
        pc.Iter(range(widths.length()))
        .map(lambda idx: row[idx].ljust(widths[idx]))
        .join(" | ")
    )
    return f"| {cells} |"


class LibCell:
    libs = pc.Seq(("narwhals", "polars"))

    def status(self, results: pc.Vec[ComparisonResult], status: Status) -> str:
        return (
            self.libs.iter()
            .map(lambda ref: _count_for_ref_status(results, ref, status))
            .collect()
            .into(lambda pair: f"({pair[0]}, {pair[1]})")
        )

    def count(
        self,
        results: pc.Vec[ComparisonResult],
        predicate: Callable[[ComparisonResult, RefBackend], bool],
    ) -> str:
        return (
            self.libs.iter()
            .map(lambda ref: _count_for_ref(results, ref, predicate))
            .collect()
            .into(lambda pair: f"({pair[0]}, {pair[1]})")
        )

    def coverage(self, results: pc.Vec[ComparisonResult]) -> str:
        return (
            self.libs.iter()
            .map(lambda ref: _coverage_percent(results, ref))
            .collect()
            .into(lambda pair: f"({pair[0]:.1f}%, {pair[1]:.1f}%)")
        )


def _coverage_percent(results: pc.Vec[ComparisonResult], ref: RefBackend) -> float:
    total = _count_for_ref(results, ref, _has_reference)
    matched = _count_for_ref_status(results, ref, Status.MATCH)
    match total:
        case 0:
            return 100.0
        case _:
            return (matched / total) * 100


def _count_for_ref(
    results: pc.Vec[ComparisonResult],
    ref: RefBackend,
    predicate: Callable[[ComparisonResult, RefBackend], bool],
) -> int:
    return results.iter().filter(lambda result: predicate(result, ref)).length()


def _count_for_ref_status(
    results: pc.Vec[ComparisonResult],
    ref: RefBackend,
    status: Status,
) -> int:
    return (
        results.iter().filter(lambda result: _has_status(result, ref, status)).length()
    )


def _reference_info(result: ComparisonResult, ref: RefBackend) -> pc.Option[MethodInfo]:
    match ref:
        case "narwhals":
            return result.infos.narwhals
        case "polars":
            return result.infos.polars


def _has_reference(result: ComparisonResult, ref: RefBackend) -> bool:
    return _reference_info(result, ref).is_some()


def _has_status(result: ComparisonResult, ref: RefBackend, status: Status) -> bool:
    return (
        _status_for_ref(result, ref)
        .map(lambda current: current == status)
        .unwrap_or(default=False)
    )


def _status_for_ref(result: ComparisonResult, ref: RefBackend) -> pc.Option[Status]:
    match (_reference_info(result, ref), result.infos.pql_info):
        case (pc.NONE, pc.NONE):
            return pc.NONE
        case (pc.Some(_), pc.NONE):
            return pc.Some(Status.MISSING)
        case (pc.NONE, pc.Some(_)):
            return pc.Some(Status.EXTRA)
        case (pc.Some(reference), pc.Some(pql_info)):
            return pc.Some(
                Status.SIGNATURE_MISMATCH
                if _mismatch_against(pql_info, reference, result.infos.ignored_params)
                else Status.MATCH
            )
        case _:
            return pc.NONE


def _mismatch_against(
    target: MethodInfo, other: MethodInfo, ignored: pc.Set[str]
) -> bool:
    target_filtered = _without_ignored_params(target.to_map(), ignored)
    other_filtered = _without_ignored_params(other.to_map(), ignored)
    on_params = (
        other_filtered.keys().symmetric_difference(target_filtered.keys()).length() > 0
    )
    on_ann = (
        other_filtered.keys()
        .intersection(target_filtered.keys())
        .any(
            lambda name: annotations_differ(
                other_filtered.get_item(name).unwrap(),
                target_filtered.get_item(name).unwrap(),
            )
        )
    )
    return on_params or on_ann


def _without_ignored_params(
    mapping: pc.Dict[str, ParamInfo], ignored: pc.Set[str]
) -> pc.Dict[str, ParamInfo]:
    return (
        mapping.items()
        .iter()
        .filter(lambda item: not ignored.contains(item[0]))
        .collect(pc.Dict)
    )


def _by_status(
    results: pc.Vec[ComparisonResult], status: Status
) -> pc.Seq[ComparisonResult]:
    return results.iter().filter(lambda r: r.classification.status == status).collect()
