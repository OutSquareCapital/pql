from dataclasses import dataclass
from typing import cast

import pyochain as pc

from ._infos import ComparisonResult, get_attr
from ._models import Status


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
                f"{_coverage_percent(self.results):.1f}%",
                str(_total_methods(self.results)),
                str(_by_status(self.results, Status.MATCH).length()),
                str(_by_status(self.results, Status.MISSING).length()),
                str(_by_status(self.results, Status.SIGNATURE_MISMATCH).length()),
                str(_by_status(self.results, Status.EXTRA).length()),
                str(_extra_vs_narwhals(self.results).length()),
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
            "Coverage vs Narwhals",
            "Total",
            "Matched",
            "Missing",
            "Mismatched",
            "Extra",
            "Extra vs Narwhals",
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
