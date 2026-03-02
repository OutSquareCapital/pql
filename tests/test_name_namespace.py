import narwhals as nw
import polars as pl

import pql

from ._utils import assert_eq, assert_eq_pl


def test_name_keep_after_alias() -> None:
    assert_eq(
        pql.col("x").alias("renamed").name.keep(),
        nw.col("x").alias("renamed").name.keep(),
    )


def test_name_map() -> None:
    assert_eq(
        pql.col("x").name.map(lambda name: f"mapped_{name}"),
        nw.col("x").name.map(lambda name: f"mapped_{name}"),
    )


def test_name_prefix_suffix() -> None:
    assert_eq(
        pql.col("x").name.prefix("pre_").name.suffix("_suf"),
        nw.col("x").name.prefix("pre_").name.suffix("_suf"),
    )


def test_name_case_transform() -> None:
    assert_eq(
        pql.col("x").name.to_uppercase().name.to_lowercase(),
        nw.col("x").name.to_uppercase().name.to_lowercase(),
    )


def test_name_to_uppercase_all() -> None:
    assert_eq(
        pql.all().name.to_uppercase(),
        nw.all().name.to_uppercase(),
    )


def test_name_replace() -> None:
    assert_eq_pl(
        pql.col("x").name.replace("x", "y"),
        pl.col("x").name.replace("x", "y"),
    )
    assert_eq_pl(
        pql.col("s").name.replace("s", "t"),
        pl.col("s").name.replace("s", "t"),
    )
    assert_eq_pl(
        pql.col("salary").name.replace("salary", "income"),
        pl.col("salary").name.replace("salary", "income"),
    )

    # Test avec literal=True (remplacement exact, pas regex)
    assert_eq_pl(
        pql.col("salary").name.replace("a", "b", literal=True),
        pl.col("salary").name.replace("a", "b", literal=True),
    )
    assert_eq_pl(
        pql.col("salary").name.replace("l", "L", literal=True),
        pl.col("salary").name.replace("l", "L", literal=True),
    )
    assert_eq_pl(
        pql.col("salary").name.replace("sal", "SAL", literal=True),
        pl.col("salary").name.replace("sal", "SAL", literal=True),
    )
