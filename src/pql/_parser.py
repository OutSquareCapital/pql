from collections.abc import Callable
from functools import cache, partial

import pyochain as pc


def _get_kwords() -> dict[str, object]:
    from sqlparse import tokens

    return {
        "PIVOT": tokens.Keyword,
        "UNPIVOT": tokens.Keyword,
        "QUALIFY": tokens.Keyword,
        "EXCLUDE": tokens.Keyword,
        "REPLACE": tokens.Keyword,
        "LAMBDA": tokens.Keyword,
    }


@cache
def _try_import() -> pc.Option[Callable[[str], str]]:  # pragma: no cover
    try:
        import sqlparse
        from sqlparse.lexer import Lexer

        Lexer.get_default_instance().add_keywords(_get_kwords())  # pyright: ignore[reportUnknownMemberType]
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
