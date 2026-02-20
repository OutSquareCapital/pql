from functools import partial
from typing import Any

import duckdb
import polars as pl
import pyochain as pc
import pytest
from polars.testing import assert_frame_equal

import pql

assert_eq = partial(assert_frame_equal, check_dtypes=False, check_row_order=False)

type TestData = dict[str, Any]


def _get_data() -> TestData:
    return {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "sex": ["F", "M", "M", "M", "F"],
        "age": [25, 30, 35, 28, 22],
        "salary": [50000.0, 60000.0, 75000.0, 55000.0, 45000.0],
        "department": [
            "Engineering",
            "Sales",
            "Engineering",
            "Sales",
            "Engineering",
        ],
        "is_active": [True, True, False, True, True],
        "value": [10.0, None, 30.0, None, 50.0],
        "category": ["A", "B", None, "A", "B"],
    }


@pytest.fixture
def data() -> TestData:
    return _get_data()


def test_from_duckdb_relation(data: TestData) -> None:
    rel = duckdb.from_arrow(pl.DataFrame(data))
    assert_eq(pql.LazyFrame(rel).collect(), rel.pl())


def test_from_table_function() -> None:
    rel = duckdb.table_function("duckdb_functions")
    assert_eq(pql.LazyFrame.from_table_function("duckdb_functions").collect(), rel.pl())


def test_from_table(data: TestData) -> None:
    duckdb.from_arrow(pl.DataFrame(data)).create("test_table")
    assert_eq(pql.LazyFrame.from_table("test_table").collect(), pl.DataFrame(data))


def test_from_pl_lazyframe(data: TestData) -> None:
    assert_eq(
        pql.LazyFrame(pl.DataFrame(data).lazy()).collect(),
        pl.DataFrame(data).lazy().collect(),
    )
    assert_eq(
        pql.LazyFrame.from_df(pl.DataFrame(data).lazy()).collect(),
        pl.DataFrame(data).lazy().collect(),
    )


def test_from_pd_dataframe(data: TestData) -> None:
    import pandas as pd

    assert_eq(
        pql.LazyFrame(pd.DataFrame(data)).collect(),
        pl.DataFrame(data),
    )
    assert_eq(
        pql.LazyFrame.from_df(pd.DataFrame(data)).collect(),
        pl.DataFrame(data),
    )


def test_from_pl_dataframe(data: TestData) -> None:
    assert_eq(
        pql.LazyFrame(pl.DataFrame(data)).collect(),
        pl.DataFrame(data),
    )
    assert_eq(
        pql.LazyFrame.from_df(pl.DataFrame(data)).collect(),
        pl.DataFrame(data),
    )


def test_from_dict(data: TestData) -> None:
    assert_eq(pql.LazyFrame(data).collect(), pl.DataFrame(data))
    assert_eq(pql.LazyFrame.from_mapping(data).collect(), pl.DataFrame(data))


def test_from_np() -> None:
    import numpy as np

    arr = np.array([[1, 2], [3, 4]])
    qry = """--sql
    SELECT *
    FROM arr"""
    rel = duckdb.from_query(qry)
    assert_eq(pql.LazyFrame(arr).collect(), rel.pl())
    assert_eq(pql.LazyFrame.from_numpy(arr).collect(), rel.pl())


def test_from_expr() -> None:
    expr = duckdb.ConstantExpression(42)
    rel = duckdb.values(expr)
    assert_eq(pql.LazyFrame(expr).collect(), rel.pl())


def test_from_pql_expr() -> None:
    expr = duckdb.ConstantExpression(42)
    rel = duckdb.values(expr)
    assert_eq(pql.LazyFrame(pql.lit(42).inner()).collect(), rel.pl())


def test_from_tup_of_exprs() -> None:
    exprs = pc.Iter(range(10)).map(duckdb.ConstantExpression).collect()
    rel = duckdb.values(exprs.into(tuple))
    assert_eq(pql.LazyFrame(exprs).collect(), rel.pl())
    assert_eq(pql.LazyFrame.from_sequence(exprs).collect(), rel.pl())


def test_from_seq_of_dicts() -> None:
    dicts = pc.Iter(range(10)).map(lambda _: _get_data()).collect()
    assert_eq(pql.LazyFrame(dicts).collect(), pl.DataFrame(dicts))
    assert_eq(pql.LazyFrame.from_sequence(dicts).collect(), pl.DataFrame(dicts))


def test_from_seq_of_seqs() -> None:
    seqs = pc.Iter(range(10)).map(lambda _: tuple(range(5))).collect()
    assert_eq(pql.LazyFrame(seqs).collect(), pl.DataFrame(seqs))
    assert_eq(pql.LazyFrame.from_sequence(seqs).collect(), pl.DataFrame(seqs))


def test_from_seq_of_vals() -> None:
    vals = pc.Iter(range(10)).map(lambda _: 42).collect()
    assert_eq(pql.LazyFrame(vals).collect(), pl.DataFrame(vals))
    assert_eq(pql.LazyFrame.from_sequence(vals).collect(), pl.DataFrame(vals))
