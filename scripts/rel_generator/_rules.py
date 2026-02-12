import pyochain as pc

PYTYPING_REWRITES: pc.Dict[str, str] = pc.Dict.from_ref(
    {
        "pytyping.Any": "Any",
        "pytyping.SupportsInt": "SupportsInt",
        "pytyping.List": "list",
        "pytyping.Literal": "Literal",
        "pytyping.Union": "Union",
        "pytorch": "torch",
        "pyarrow.lib.": "pa.",
        "pyarrow.": "pa.",
    }
)

TYPE_SUBS: pc.Dict[str, str] = pc.Dict.from_kwargs(
    Expression="SqlExpr",
    DuckDBPyRelation="Self",
)

SKIP_METHODS: pc.Set[str] = pc.Set(
    {
        "__arrow_c_stream__",
        "__contains__",
        "__getattr__",
        "__getitem__",
        "__len__",
        "close",
        "execute",
        "map",
    }
)
