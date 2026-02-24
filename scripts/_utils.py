from __future__ import annotations

import collections.abc as collections_abc
import datetime
import decimal
import typing as ty
from enum import StrEnum, auto

import duckdb
import pyochain as pc

import pql


class KwordEnum(StrEnum):
    """Enum to easily manipulate python constructs as text, e.g. for code generation."""

    @classmethod
    def module(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def into_iter(cls) -> pc.Iter[KwordEnum]:
        return pc.Iter(cls)


def get_attr(obj: object, name: str) -> pc.Option[object]:
    """Safe getattr returning Option."""
    return pc.Option(getattr(obj, name, None))


class Dunders(KwordEnum):
    INIT = "__init__"
    CALL = "__call__"
    DEPRECATED = "__deprecated__"
    DOC = "__doc__"


class Pql(KwordEnum):
    SQLEXPR = "SqlExpr"
    EXPR = pql.Expr.__name__
    TRY_ITER = auto()
    INTO_DUCKDB = auto()
    RELATION = "Relation"
    CORE_HANDLER = "CoreHandler"
    REL_HANDLER = "RelHandler"
    DUCK_HANDLER = "DuckHandler"

    LAZY_FRAME = pql.LazyFrame.__name__
    LAZY_GROUP_BY = "LazyGroupBy"
    EXPR_STR_NAME_SPACE = "ExprStrNameSpace"
    EXPR_LIST_NAME_SPACE = "ExprListNameSpace"
    EXPR_STRUCT_NAME_SPACE = "ExprStructNameSpace"
    EXPR_NAME_NAME_SPACE = "ExprNameNameSpace"
    EXPR_ARR_NAME_SPACE = "ExprArrNameSpace"


class Pyochain(KwordEnum):
    OPTION = pc.Option.__name__
    SEQ = pc.Seq.__name__
    ITER = pc.Iter.__name__
    VEC = pc.Vec.__name__
    DICT = pc.Dict.__name__
    SET = pc.Set.__name__


class Builtins(KwordEnum):
    NONE = "None"
    LIST = list.__name__
    DICT = dict.__name__
    PROPERTY = property.__name__
    SELF = auto()
    STR = str.__name__
    BOOL = bool.__name__
    INT = int.__name__
    FLOAT = float.__name__
    BYTES = bytes.__name__
    BYTEARRAY = bytearray.__name__
    MEMORYVIEW = memoryview.__name__


class DateTime(KwordEnum):
    TIME = datetime.time.__name__
    DATE = datetime.date.__name__
    DATETIME = datetime.datetime.__name__
    TIMEDELTA = datetime.timedelta.__name__


class DuckDB(KwordEnum):
    EXPRESSION = duckdb.Expression.__name__
    RELATION = duckdb.DuckDBPyRelation.__name__


class Decimal(KwordEnum):
    DECIMAL = decimal.Decimal.__name__


class Typing(KwordEnum):
    ANY = ty.Any.__name__
    LITERAL = ty.Literal.__name__
    SELF = ty.Self.__name__
    UNION = ty.Union.__name__  # pyright: ignore[reportDeprecated]
    SUPPORTS_INT = ty.SupportsInt.__name__
    OVERLOAD = ty.overload.__name__


class CollectionsABC(KwordEnum):
    @classmethod
    def module(cls) -> str:
        return "collections.abc"

    ITERABLE = collections_abc.Iterable.__name__
    CALLABLE = collections_abc.Callable.__name__
