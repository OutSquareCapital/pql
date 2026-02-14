from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple, Self

import pyochain as pc

from ._models import MismatchOn, Status

type MapInfo = pc.Dict[str, ParamInfo]

SELF_PATTERN = re.compile(r"\b(Self|Expr|LazyFrame)\b")
GENERIC_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


class _GenericCanonicalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._mapping: dict[str, str] = {}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        match node.id:
            case name if name not in {"Self", "Expr", "LazyFrame"} and bool(
                GENERIC_SYMBOL_PATTERN.match(name)
            ):
                canonical_name = self._mapping.setdefault(
                    name, f"__GENERIC_{len(self._mapping)}__"
                )
                return ast.copy_location(
                    ast.Name(id=canonical_name, ctx=node.ctx), node
                )
            case _:
                return node


def annotations_differ(pl_param: ParamInfo, pql_param: ParamInfo) -> bool:
    match (pl_param.annotation, pql_param.annotation):
        case (pc.Some(pl_ann), pc.Some(pql_ann)):
            return _normalize_self(
                _extract_last_name(_normalize_generics(pl_ann))
            ) != _normalize_self(_extract_last_name(_normalize_generics(pql_ann)))
        case _:
            return False


class MethodStatus(NamedTuple):
    """Status of a method comparison."""

    status: Status
    mismatch_source: MismatchOn


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


def _get_annotation_str(annotation: object) -> pc.Option[str]:
    """Convert annotation to string representation."""
    match annotation:
        case inspect.Parameter.empty | inspect.Signature.empty:
            return pc.NONE
        case type():
            return pc.Option(annotation.__name__)
        case _:
            return pc.Option(_extract_last_name(str(annotation)))


def _mismatch_against(target: MapInfo, other: MapInfo) -> bool:
    on_params = other.keys().symmetric_difference(target.keys()).length() > 0
    on_ann = (
        other.keys()
        .intersection(target.keys())
        .any(
            lambda name: annotations_differ(
                other.get_item(name).unwrap(), target.get_item(name).unwrap()
            )
        )
    )
    return on_params or on_ann


def _normalize_generics(annotation: str) -> str:
    try:
        parsed = ast.parse(annotation, mode="eval")
    except SyntaxError:
        return annotation

    normalized = _GenericCanonicalizer().visit(parsed)
    ast.fix_missing_locations(normalized)
    return ast.unparse(normalized)


def _normalize_self(annotation: str) -> str:
    return _extract_last_name(SELF_PATTERN.sub("__SELF__", annotation))


def _extract_last_name(annotation: str) -> str:
    if "[" in annotation:
        base_type = annotation.split("[", maxsplit=1)[0]
        generic_part = annotation[len(base_type) :]
        return _extract_last_name(base_type) + generic_part

    return annotation.rsplit(".", maxsplit=1)[-1]


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


def _get_method_info(cls: type, name: str) -> pc.Option[MethodInfo]:
    return get_attr(cls, name).and_then(_build_method_info, name)


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


def get_attr(obj: object, name: str) -> pc.Option[object]:
    """Safe getattr returning Option."""
    return pc.Option(getattr(obj, name, None))


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
                lambda name: annotations_differ(
                    base_map.get_item(name).unwrap(), other_map.get_item(name).unwrap()
                )
            )
        )
        .collect(pc.Set)
    )
