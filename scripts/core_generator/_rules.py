import pyochain as pc

from .._utils import Builtins, Typing

PYTYPING_REWRITES = pc.Dict.from_ref(
    {
        "typing.Any": Typing.ANY,
        "typing.SupportsInt": Typing.SUPPORTS_INT,
        "typing.Literal": Typing.LITERAL,
        "pyarrow.lib.": "pa.",
        "lst": Builtins.LIST,
    }
)
