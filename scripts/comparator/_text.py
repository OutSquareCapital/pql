from collections.abc import Callable
from dataclasses import dataclass

import pyochain as pc

from ._infos import (
    ComparisonResult,
    RefBackend,
    get_attr,
)
from ._models import Status

REF_BACKENDS = pc.Seq(("narwhals", "polars"))


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
        return pc.Seq(
            (
                self.name,
                _coverage_cell(self.results),
                _count_cell(self.results, _has_reference),
                _status_cell(self.results, Status.MATCH),
                _status_cell(self.results, Status.MISSING),
                _status_cell(self.results, Status.SIGNATURE_MISMATCH),
                _status_cell(self.results, Status.EXTRA),
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

    narwhals_cls: pc.Option[type]
    polars_cls: type
    pql_cls: type
    name: str

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
            self.narwhals_cls.map(_get_public_methods)
            .unwrap_or(pc.Set[str].new())
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


def _for_each_ref[T](mapper: Callable[[RefBackend], T]) -> pc.Seq[T]:
    return REF_BACKENDS.iter().map(mapper).collect()


def _count_cell(
    results: pc.Vec[ComparisonResult],
    predicate: Callable[[ComparisonResult, RefBackend], bool],
) -> str:
    return _for_each_ref(lambda ref: _count_for_ref(results, ref, predicate)).into(
        lambda pair: f"({pair[0]}, {pair[1]})"
    )


def _status_cell(results: pc.Vec[ComparisonResult], status: Status) -> str:
    return _for_each_ref(lambda ref: _count_for_ref_status(results, ref, status)).into(
        lambda pair: f"({pair[0]}, {pair[1]})"
    )


def _coverage_cell(results: pc.Vec[ComparisonResult]) -> str:
    return _for_each_ref(lambda ref: _coverage_percent(results, ref)).into(
        lambda pair: f"({pair[0]:.1f}%, {pair[1]:.1f}%)"
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
    results: pc.Vec[ComparisonResult], ref: RefBackend, status: Status
) -> int:
    return (
        results.iter()
        .filter(
            lambda result: (
                result.infos.status_for_ref(ref)
                .map(lambda current: current == status)
                .unwrap_or(default=False)
            )
        )
        .length()
    )


def _has_reference(result: ComparisonResult, ref: RefBackend) -> bool:
    return result.infos.has_reference(ref)


def _by_status(
    results: pc.Vec[ComparisonResult], status: Status
) -> pc.Seq[ComparisonResult]:
    return results.iter().filter(lambda r: r.classification.status == status).collect()
