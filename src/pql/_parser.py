from __future__ import annotations

from collections.abc import Callable
from functools import cache, partial
from typing import TYPE_CHECKING

import pyochain as pc

if TYPE_CHECKING:
    from ._typing import Themes


@cache
def _get_kwords():  # noqa: ANN202
    from sqlparse.tokens import Keyword

    from ._creation import from_table_function
    from .sql import col, lit, when

    name = col("keyword_name")

    return (
        from_table_function("duckdb_keywords")
        .inner()
        .select(
            when(col("keyword_category").is_in(lit("reserved"), lit("unreserved")))
            .then(name.str.upper())
            .otherwise(name)
        )
        .fetchall()
        .iter()
        .flatten()
        .map(lambda x: (x, Keyword))  # pyright: ignore[reportAny]
        .collect(dict)
    )


@cache
def _formatter() -> partial[str]:  # pragma: no cover
    import sqlparse
    from sqlparse.lexer import Lexer

    Lexer.get_default_instance().add_keywords(_get_kwords())  # pyright: ignore[reportUnknownMemberType]
    return partial(
        sqlparse.format,
        indent_width=4,
        reindent=True,
        keyword_case="upper",
        use_space_around_operators=True,
    )


def _try_import[T](importer: Callable[[], T]) -> pc.Option[T]:  # pragma: no cover
    try:
        return pc.Some(importer())
    except ImportError:
        return pc.NONE


@cache
def _colorizer() -> Callable[[str, Themes], None]:  # pragma: no cover
    from rich.console import Console
    from rich.syntax import Syntax

    def _printer(txt: str, theme: Themes) -> None:
        return Console().print(
            Syntax(txt, lexer="sql", theme=theme, background_color="default")
        )

    return _printer


def format_sql(qry: str) -> str:
    return _try_import(_formatter).map(lambda sp: sp(qry)).unwrap_or(qry)


def show_sql(qry: str, theme: Themes) -> None:  # pragma: no cover
    return (
        _try_import(_colorizer)
        .map(lambda printer: printer(qry, theme))
        .unwrap_or_else(lambda: print(qry))
    )
