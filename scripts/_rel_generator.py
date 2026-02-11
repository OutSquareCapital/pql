import inspect
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

import duckdb
import pyochain as pc
import rich.console

console = rich.console.Console()


def _get_attr(obj: object, attr: str) -> pc.Option[Any]:
    return pc.Option(getattr(obj, attr, None))


class Attrs(StrEnum):
    NAME = "__name__"
    QUALNAME = "__qualname__"
    TEXT_SIGNATURE = "__text_signature__"


@dataclass(slots=True)
class _MemberInfo:
    name: str
    doc: str
    signature: str

    @classmethod
    def from_member(cls, member: object) -> pc.Option[Self]:
        return (
            _get_attr(member, Attrs.NAME)
            .or_else(lambda: _get_attr(member, Attrs.QUALNAME))
            .filter(lambda name: not name.startswith("_"))
            .zip_with(
                pc.Option(inspect.getdoc(member)).or_else(lambda: pc.Some("")),
                lambda name, doc: cls(
                    name=name,
                    doc=_strip_signature_from_doc(name, doc),
                    signature=_get_signature(member, name, doc),
                ),
            )
        )

    def format(self) -> str:
        return (
            pc.Vec.from_ref(self.doc.splitlines())
            .then_some()
            .map(
                lambda lines: (
                    f'    """{lines.first()}."""'
                    if lines.length() == 1
                    else f'    """{lines.iter().map(lambda line: f"    {line}").join("\n")}\n    """'
                )
            )
            .map(lambda formatted: f"def {self.signature}:\n{formatted}")
            .unwrap_or(f"def {self.signature}:")
        )


def _get_signature(member: object, name: str, doc: str) -> str:

    def _safe_signature(member: Any) -> pc.Result[str, Exception]:  # noqa: ANN401
        try:
            return pc.Ok(str(inspect.signature(member)))
        except (TypeError, ValueError) as exc:
            return pc.Err(exc)

    return (
        _safe_signature(member)
        .ok()
        .or_else(lambda: _get_attr(member, Attrs.TEXT_SIGNATURE))
        .map(lambda sig: f"{name}{sig}")
        .or_else(
            lambda: (
                pc.Vec.from_ref(doc.splitlines())
                .then_some()
                .map(lambda xs: xs.first().strip())
                .filter(lambda first: first.startswith(f"{name}("))
            )
        )
        .expect(f"Could not determine signature for member {name!r}")
    )


def _strip_signature_from_doc(name: str, doc: str) -> str:
    return (
        pc.Vec.from_ref(doc.splitlines())
        .then_some()
        .filter(lambda lines: lines.first().strip().startswith(f"{name}("))
        .map(lambda lines: lines.iter().skip(1).skip_while(lambda line: line == ""))
        .map(lambda remaining: remaining.join("\n"))
        .unwrap_or(doc)
    )


def main(obj: object) -> None:
    return (
        pc.Iter(inspect.getmembers(obj, predicate=inspect.isroutine))
        .filter_map_star(lambda _, member: _MemberInfo.from_member(member))
        .map(lambda info: info.format())
        .into(lambda x: rich.print(x.join("\n\n")))
    )


if __name__ == "__main__":
    console.print("Expression:", style="bold magenta")
    main(duckdb.Expression)
    console.print("DuckDBPyRelation:", style="bold magenta")
    main(duckdb.DuckDBPyRelation)
