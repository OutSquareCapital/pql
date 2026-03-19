from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

import pyochain as pc

from ._infos import ComparisonResult
from ._rules import RefBackend, Status


class BarColor(StrEnum):
    RED = "#e74c3c"
    ORANGE = "#f39c12"
    GREEN = "#27ae60"

    @classmethod
    def on_pct(cls, percentage: float) -> str:
        match percentage:
            case p if p < 30:
                return cls.RED
            case p if p < 60:
                return cls.ORANGE
            case _:
                return cls.GREEN


@dataclass(slots=True)
class ArrayBuilder:
    results: pc.Vec[ComparisonResult]
    _row: pc.Vec[str] = field(default_factory=pc.Vec.new)

    def build(self) -> pc.Vec[str]:
        return self._row

    def _add(self, element: str) -> Self:
        self._row.append(element)
        return self

    def count_cell(self) -> Self:
        return self._add(
            _for_each_ref(
                lambda ref: _count_for_ref(self.results, ref, _has_reference)
            ).into(_int_pair_with_total)
        )

    def status_cell(self, status: Status) -> Self:
        return self._add(
            _for_each_ref(
                lambda ref: _count_for_ref_status(self.results, ref, status)
            ).into(_int_pair_with_total)
        )

    def coverage_cell(self) -> Self:
        global_pct = _global_coverage_percent(self.results)
        bar = _generate_progress_bar(global_pct)
        return self._add(
            _for_each_ref(lambda ref: _coverage_percent(self.results, ref)).into(
                lambda pair: f"{bar} ({pair[0]:.1f}%, {pair[1]:.1f}%)"
            )
        )

    def with_name(self, name: str) -> Self:
        return self._add(name)


def _has_reference(result: ComparisonResult, ref: RefBackend) -> bool:
    return result.infos.has_reference(ref)


def _global_coverage_percent(results: pc.Vec[ComparisonResult]) -> float:
    totals = _for_each_ref(lambda ref: _count_for_ref(results, ref, _has_reference))
    matched = _for_each_ref(
        lambda ref: _count_for_ref_status(results, ref, Status.MATCH)
    )
    total = totals.sum()
    match total:
        case 0:
            return 100.0
        case _:
            return (matched.sum() / total) * 100


def _coverage_percent(results: pc.Vec[ComparisonResult], ref: RefBackend) -> float:
    total = _count_for_ref(results, ref, _has_reference)
    matched = _count_for_ref_status(results, ref, Status.MATCH)
    match total:
        case 0:
            return 100.0
        case _:
            return (matched / total) * 100


def _for_each_ref[T](mapper: Callable[[RefBackend], T]) -> pc.Seq[T]:
    return pc.Iter(RefBackend).map(mapper).collect()


def _generate_progress_bar(percentage: float, width: int = 10) -> str:
    """Generate a Unicode progress bar with color based on percentage."""
    filled = int(percentage / 100 * width)
    empty = width - filled
    color = BarColor.on_pct(percentage)

    filled_bar = f'<span style="color: {color};">{"█" * filled}</span>'
    empty_bar = f'<span style="color: #bdc3c7;">{"░" * empty}</span>'
    return filled_bar + empty_bar


def _int_pair_with_total(pair: pc.Seq[int]) -> str:
    return f"{pair.sum()} ({pair[0]}, {pair[1]})"


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
