from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Self

import pyochain as pc

from ._models import MismatchOn, Status
from ._parse import extract_last_name, normalize_annotation

type MapInfo = pc.Dict[str, ParamInfo]
type IgnoredParams = pc.Dict[str, pc.Dict[str, pc.Set[str]]]
type RefBackend = Literal["narwhals", "polars"]

IGNORED_PARAMS_BY_CLASS_AND_METHOD: IgnoredParams = pc.Dict.from_kwargs(
    LazyFrame=pc.Dict.from_kwargs(
        sort=pc.Set(("maintain_order", "multithreaded")),
    )
)


def annotations_differ(pl_param: ParamInfo, pql_param: ParamInfo) -> bool:
    match (pl_param.annotation, pql_param.annotation):
        case (pc.Some(pl_ann), pc.Some(pql_ann)):
            normalized_pl = normalize_annotation(pl_ann)
            normalized_pql = normalize_annotation(pql_ann)
            if normalized_pl == "Any":
                return False
            return normalized_pl != normalized_pql
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
    is_var_positional: bool
    is_var_keyword: bool
    has_default: bool
    annotation: pc.Option[str]

    @classmethod
    def from_signature(cls, param: inspect.Parameter) -> Self:
        """Create ParamInfo from inspect.Parameter."""
        return cls(
            name=param.name,
            is_var_positional=param.kind == inspect.Parameter.VAR_POSITIONAL,
            is_var_keyword=param.kind == inspect.Parameter.VAR_KEYWORD,
            has_default=param.default is not inspect.Parameter.empty,
            annotation=_get_annotation_str(param.annotation),
        )

    def param_name(self) -> str:
        match (self.is_var_positional, self.is_var_keyword):
            case (True, _):
                return f"*{self.name}"
            case (_, True):
                return f"**{self.name}"
            case _:
                return self.name


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
                lambda p: p.annotation.map(
                    lambda a: f"{p.param_name()}: {a}"
                ).unwrap_or(p.param_name() + ("=..." if p.has_default else ""))
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
    ignored_params: pc.Set[str] = field(default_factory=pc.Set[str].new)

    def reference(self, ref: RefBackend) -> pc.Option[MethodInfo]:
        match ref:
            case "narwhals":
                return self.narwhals
            case "polars":
                return self.polars

    def has_reference(self, ref: RefBackend) -> bool:
        return self.reference(ref).is_some()

    def status_for_ref(self, ref: RefBackend) -> pc.Option[Status]:
        match (self.reference(ref), self.pql_info):
            case (pc.NONE, pc.NONE):
                return pc.NONE
            case (pc.Some(_), pc.NONE):
                return pc.Some(Status.MISSING)
            case (pc.NONE, pc.Some(_)):
                return pc.Some(Status.EXTRA)
            case (pc.Some(reference), pc.Some(pql_info)):
                return pc.Some(
                    Status.SIGNATURE_MISMATCH
                    if _mismatch_against(
                        pql_info.to_map(),
                        reference.to_map(),
                        self.ignored_params,
                    )
                    else Status.MATCH
                )
            case _:
                return pc.NONE

    def to_status(self) -> MethodStatus:  # noqa: C901, PLR0911
        """Classify the method comparison result."""
        match (self.pql_info, self.narwhals, self.polars):
            case (pc.NONE, pc.Some(_), _):
                return MethodStatus(Status.MISSING, MismatchOn.NULL)
            case (pc.Some(_), pc.NONE, pc.NONE):
                return MethodStatus(Status.EXTRA, MismatchOn.NULL)
            case (pc.Some(target), pc.NONE, pc.Some(pl_info)):
                match _method_mismatch(target, pl_info, self.ignored_params):
                    case True:
                        return MethodStatus(Status.SIGNATURE_MISMATCH, MismatchOn.PL)
                    case False:
                        return MethodStatus(Status.MATCH, MismatchOn.NULL)
            case (pc.Some(target), pc.Some(nw_info), pc.Some(pl_info)):
                nw_mismatch = _method_mismatch(target, nw_info, self.ignored_params)
                pl_mismatch = _method_mismatch(target, pl_info, self.ignored_params)
                match nw_mismatch and pl_mismatch:
                    case True:
                        return MethodStatus(Status.SIGNATURE_MISMATCH, MismatchOn.NW)
                    case False:
                        return MethodStatus(Status.MATCH, MismatchOn.NULL)
            case (pc.Some(target), pc.Some(nw_info), pc.NONE):
                match _method_mismatch(target, nw_info, self.ignored_params):
                    case True:
                        return MethodStatus(Status.SIGNATURE_MISMATCH, MismatchOn.NW)
                    case False:
                        return MethodStatus(Status.MATCH, MismatchOn.NULL)
            case _:
                return MethodStatus(Status.MISSING, MismatchOn.NULL)


def _method_mismatch(
    target: MethodInfo, other: MethodInfo, ignored: pc.Set[str]
) -> bool:
    return _mismatch_against(target.to_map(), other.to_map(), ignored)


def _get_annotation_str(annotation: object) -> pc.Option[str]:
    """Convert annotation to string representation."""
    match annotation:
        case inspect.Parameter.empty | inspect.Signature.empty:
            return pc.NONE
        case type():
            return pc.Option(annotation.__name__)
        case _:
            return pc.Option(extract_last_name(str(annotation)))


def _mismatch_against(target: MapInfo, other: MapInfo, ignored: pc.Set[str]) -> bool:
    target_filtered = _without_ignored_params(target, ignored)
    other_filtered = _without_ignored_params(other, ignored)
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


def _without_ignored_params(mapping: MapInfo, ignored: pc.Set[str]) -> MapInfo:
    return (
        mapping.items()
        .iter()
        .filter(lambda item: not ignored.contains(item[0]))
        .collect(pc.Dict)
    )


def ignored_params_for(class_name: str, method_name: str) -> pc.Set[str]:
    return (
        IGNORED_PARAMS_BY_CLASS_AND_METHOD.get_item(class_name)
        .and_then(lambda method_map: method_map.get_item(method_name))
        .unwrap_or(default=pc.Set[str].new())
    )


@dataclass(slots=True)
class ComparisonResult:
    """Result of comparing a single method."""

    method_name: str
    classification: MethodStatus
    infos: ComparisonInfos

    @classmethod
    def from_method(
        cls,
        narwhals_cls: pc.Option[type],
        polars_cls: type,
        pql_cls: type,
        method_name: str,
        class_name: str,
    ) -> Self:
        """Compare a single method between narwhals, polars, and pql."""
        infos = ComparisonInfos(
            narwhals=narwhals_cls.and_then(_get_method_info, method_name),
            polars=_get_method_info(polars_cls, method_name),
            pql_info=_get_method_info(pql_cls, method_name),
            ignored_params=ignored_params_for(class_name, method_name),
        )
        return cls(
            method_name=method_name,
            classification=infos.to_status(),
            infos=infos,
        )

    def to_format(self, *, status: Status) -> pc.Iter[str]:
        """Format a single comparison result as markdown lines."""
        match (status, self.infos.narwhals, self.infos.polars, self.infos.pql_info):
            case (Status.MISSING, _, _, _):
                return (
                    pc.Iter.once(f"- `{self.method_name}`")
                    .chain(
                        self.infos.narwhals.map(
                            lambda info: pc.Iter.once(
                                f"  - **Narwhals**: {info.signature_str()}"
                            )
                        ).unwrap_or(default=pc.Iter(()))
                    )
                    .chain(
                        self.infos.polars.map(
                            lambda info: pc.Iter.once(
                                f"  - **Polars**: {info.signature_str()}"
                            )
                        ).unwrap_or(default=pc.Iter(()))
                    )
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
                            f"  - **Narwhals**: {_signature_with_diff(nw_info, pql_info, self.infos.ignored_params)}",
                        )
                    )
                    .chain(
                        self.infos.polars.map(
                            lambda pl_info: pc.Iter.once(
                                f"  - **Polars**: {_signature_with_diff(pl_info, pql_info, self.infos.ignored_params)}"
                            )
                        ).unwrap_or(default=pc.Iter(()))
                    )
                    .chain(
                        pc.Iter.once(
                            f"  - **pql**: {_signature_with_diff(pql_info, nw_info, self.infos.ignored_params)}"
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
                        f"  - **Polars***: {_signature_with_diff(pl_info, pql_info, self.infos.ignored_params)}",
                        f"  - **pql**: {_signature_with_diff(pql_info, pl_info, self.infos.ignored_params)}",
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


def is_deprecated_method(obj: object, name: str) -> bool:
    return (
        get_attr(obj, name)
        .map(lambda attr: bool(getattr(attr, "__deprecated__", False)))
        .unwrap_or(default=False)
    )


def _signature_with_diff(
    base: MethodInfo, other: MethodInfo, ignored: pc.Set[str]
) -> str:
    diff_names = _diff_param_names(base, other, ignored)

    def _format_param(p: ParamInfo) -> str:
        match diff_names.any(lambda name: name == p.name):
            case True:
                return f"`{p.annotation.map(lambda a: f'{p.param_name()}: {a}').unwrap_or(p.param_name() + ('=...' if p.has_default else ''))}`"
            case False:
                return p.annotation.map(lambda a: f"{p.param_name()}: {a}").unwrap_or(
                    p.param_name() + ("=..." if p.has_default else "")
                )

    params_str = (
        base.params.iter()
        .filter(lambda p: p.name != "self")
        .map(_format_param)
        .join(", ")
    )
    ret = base.return_annotation.map(lambda r: f" -> {r}").unwrap_or("")
    return f"({params_str}){ret}"


def _diff_param_names(
    base: MethodInfo, other: MethodInfo, ignored: pc.Set[str]
) -> pc.Set[str]:
    base_map = _without_ignored_params(base.to_map(), ignored)
    other_map = _without_ignored_params(other.to_map(), ignored)
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
