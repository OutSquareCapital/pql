import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Self

import pyochain as pc
import sqlglot
from sqlglot.errors import ParseError

_POINTER_LITERAL_PATTERN = re.compile(r"\b0x[0-9a-fA-F]+\b")
_POINTER_FN_CALL_PATTERN = re.compile(
    r"(?P<fn>\b(?:arrow_scan|arrow_scan_dumb|pandas_scan|python_map_function)\s*\()(?P<args>[^)]*)(?P<end>\))"
)
_POINTER_PLACEHOLDER_PREFIX: Final[str] = "__pql_pointer_"


class Kword(StrEnum):
    PARTITION_BY = "PARTITION BY"
    ORDER_BY = "ORDER BY"
    DESC = "DESC"
    ASC = "ASC"
    NULLS_LAST = "NULLS LAST"
    NULLS_FIRST = "NULLS FIRST"
    ROWS_BETWEEN = "ROWS BETWEEN"
    OVER = "OVER"
    SELECT = "SELECT"
    FROM = "FROM"
    WHERE = "WHERE"
    GROUP_BY = "GROUP BY"
    HAVING = "HAVING"
    JOIN = "JOIN"
    LEFT_JOIN = "LEFT JOIN"
    RIGHT_JOIN = "RIGHT JOIN"
    INNER_JOIN = "INNER JOIN"
    FULL_JOIN = "FULL JOIN"
    ON = "ON"
    AND = "AND"
    OR = "OR"
    LIMIT = "LIMIT"
    OFFSET = "OFFSET"

    @classmethod
    def rows_clause(cls, row_start: pc.Option[int], row_end: pc.Option[int]) -> str:
        match (row_start, row_end):
            case (pc.Some(start), pc.Some(end)):
                return f"{cls.ROWS_BETWEEN} {-start} PRECEDING AND {end} FOLLOWING"
            case (pc.Some(start), pc.NONE):
                return f"{cls.ROWS_BETWEEN} {-start} PRECEDING AND UNBOUNDED FOLLOWING"
            case (pc.NONE, pc.Some(end)):
                return f"{cls.ROWS_BETWEEN} UNBOUNDED PRECEDING AND {end} FOLLOWING"
            case _:
                return ""

    @classmethod
    def partition_by(cls, by: str) -> str:
        return f"{cls.PARTITION_BY} {by}"

    @classmethod
    def order_by(cls, by: str) -> str:
        return f"{cls.ORDER_BY} {by}"


@dataclass(slots=True)
class QueryHolder:
    query: str
    pointer_map: pc.Dict[str, str]

    @classmethod
    def from_query(cls, qry: str) -> Self:
        pointer_map = pc.Dict[str, str].new()

        def _sanitize_pointer_literal(match: re.Match[str]) -> str:
            placeholder = f"'{_POINTER_PLACEHOLDER_PREFIX}{pointer_map.length()}__'"
            pointer_map[placeholder] = match.group(0)
            return placeholder

        def _replace_pointers(match: re.Match[str]) -> str:
            return (
                f"{match.group('fn')}"
                f"{_POINTER_LITERAL_PATTERN.sub(_sanitize_pointer_literal, match.group('args'))}"
                f"{match.group('end')}"
            )

        return cls(_POINTER_FN_CALL_PATTERN.sub(_replace_pointers, qry), pointer_map)

    def restore(self, qry: str) -> str:
        return (
            self.pointer_map.items()
            .iter()
            .fold(qry, lambda current_qry, item: current_qry.replace(*item))
        )

    def prettify(self) -> str:
        try:
            return self.restore(
                sqlglot.parse_one(self.query, dialect="duckdb").sql(  # pyright: ignore[reportUnknownMemberType]
                    dialect="duckdb", pretty=True
                )
            )
        except ParseError:
            return self.restore(self.query)
