from collections.abc import Callable

import pyochain as pc


def _try_import() -> pc.Option[Callable[[str], str]]:  # pragma: no cover
    from functools import partial

    try:
        import sqlparse

        fn = partial(
            sqlparse.format,
            indent_width=4,
            reindent=True,
            keyword_case="upper",
            use_space_around_operators=True,
        )

        return pc.Some(fn)
    except ImportError:
        return pc.NONE


def format_sql(qry: str) -> str:
    return _try_import().map(lambda sp: sp(qry)).unwrap_or(qry)
