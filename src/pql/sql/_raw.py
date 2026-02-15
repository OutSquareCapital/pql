import re
from enum import StrEnum

import pyochain as pc
import sqlglot

_POINTER_LITERAL_PATTERN = re.compile(r"\b0x[0-9a-fA-F]+\b")
_POINTER_FN_CALL_PATTERN = re.compile(
    r"(?P<fn>\b(?:arrow_scan|arrow_scan_dumb|pandas_scan|python_map_function)\s*\()(?P<args>[^)]*)(?P<end>\))"
)


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

    @classmethod
    def prettify(cls, qry: str) -> str:
        return sqlglot.parse_one(to_sqlglot(qry), dialect="duckdb").sql(
            dialect="duckdb", pretty=True
        )


def to_sqlglot(qry: str) -> str:
    def _replace_pointers(match: re.Match[str]) -> str:
        return (
            f"{match.group('fn')}"
            f"{_POINTER_LITERAL_PATTERN.sub('pointer', match.group('args'))}"
            f"{match.group('end')}"
        )

    return _POINTER_FN_CALL_PATTERN.sub(_replace_pointers, qry)
