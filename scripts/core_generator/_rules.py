import pyochain as pc

from .._utils import Builtins, Dunders, Typing

PYTYPING_REWRITES = pc.Dict.from_ref(
    {
        "typing.Any": Typing.ANY,
        "typing.SupportsInt": Typing.SUPPORTS_INT,
        "typing.Literal": Typing.LITERAL,
        "pyarrow.lib.": "pa.",
        "lst": Builtins.LIST,
    }
)

DUNDER_OPERATOR_ALIAS_EXCEPTIONS: pc.Dict[str, str] = pc.Dict.from_ref(
    {Dunders.AND: "and_", Dunders.OR: "or_", Dunders.INVERT: "not_"}
)


def dunder_operator_alias(name: str) -> pc.Option[str]:
    return DUNDER_OPERATOR_ALIAS_EXCEPTIONS.get_item(name).or_else(
        lambda: pc.Option.if_true(
            name,
            predicate=lambda x: (
                x.startswith("__") and x.endswith("__") and x != Dunders.INIT
            ),
        ).map(lambda x: x.removeprefix("__").removesuffix("__"))
    )
